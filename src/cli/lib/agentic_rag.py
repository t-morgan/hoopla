"""Agentic Recursive RAG implementation.

This module implements an agentic search system that can dynamically choose
and combine different search tools based on the query and previous results.
"""

import json
import logging
import uuid
from typing import Any
from dataclasses import dataclass, field

from .llm_utils import execute_llm_prompt
from .search_utils import load_movies

from .agentic_tools import (
    GENRE_SYNONYMS,
    extract_json_object,
    KeywordSearchTool,
    SemanticSearchTool,
    HybridSearchTool,
    RegexSearchTool,
    GenreSearchTool,
    ActorSearchTool,
)

logger = logging.getLogger(__name__)


@dataclass
class SearchResult:
    """Container for search results from a tool."""
    tool_name: str
    query: str
    results: list[dict[str, Any]]
    reasoning: str = ""
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class AgenticSearchConfig:
    """Configuration for agentic search."""
    max_iterations: int = 5
    max_results_per_tool: int = 10
    final_result_limit: int = 5
    debug: bool = False
    min_intersection_matches: int = 2
    intersection_mode: str = "auto"  # "strict", "loose", "auto"




class AgenticRAG:
    """Agentic RAG system that dynamically chooses search tools."""

    def __init__(
        self,
        config: AgenticSearchConfig | None = None,
        movies: list[dict[str, Any]] | None = None,
    ):
        self.config = config or AgenticSearchConfig()
        self.movies = movies or load_movies()
        self.tools = {
            'semantic_search': SemanticSearchTool(),
            'hybrid_search': HybridSearchTool(self.movies),
            'keyword_search': KeywordSearchTool(),
            'regex_search': RegexSearchTool(self.movies),
            'genre_search': GenreSearchTool(self.movies),
            'actor_search': ActorSearchTool(self.movies),
        }
        # Removed instance-level search_history and candidate_pool for statelessness

    def _build_history_context(self, previous_results: list[SearchResult]) -> str:
        """Build a readable history context including sample docs for the LLM."""
        if not previous_results:
            return ""

        lines: list[str] = ["Previous searches:"]
        for i, result in enumerate(previous_results, 1):
            lines.append(
                f"{i}. {result.tool_name} with query '{result.query}': {len(result.results)} results"
            )
            if result.reasoning:
                lines.append(f"   Reasoning: {result.reasoning}")

            # Show a few example docs so the LLM can reason about them
            for movie in result.results[:3]:
                title = movie.get("title", "Unknown title")
                desc = (movie.get("description", "") or "")[:160].replace("\n", " ")
                lines.append(f"   - {title}: {desc}...")

        text = "\n".join(lines)
        # Truncate to prevent context overflow (roughly 1000 tokens = 4000 chars)
        return text[:4000]

    def _build_candidate_summary(self, candidate_pool: dict[int, dict[str, Any]]) -> str:
        """Summarize current candidate pool for the LLM."""
        if not candidate_pool:
            return "No candidates accumulated yet."

        movies = list(candidate_pool.values())
        movies.sort(key=lambda m: m.get("score", 0), reverse=True)
        lines = ["Current candidate movies:"]
        for i, m in enumerate(movies[:5], 1):
            title = m.get("title", "Unknown title")
            desc = (m.get("description", "") or "")[:160].replace("\n", " ")
            lines.append(f"[{i}] {title}: {desc}...")
        text = "\n".join(lines)
        # Truncate to prevent context overflow (roughly 750 tokens = 3000 chars)
        return text[:3000]

    def _pick_next_tool(
        self,
        query: str,
        previous_results: list[SearchResult],
        used_pairs: set[tuple[str, str]],
        candidate_pool: dict[int, dict[str, Any]]
    ) -> tuple[str, str, str] | None:
        """Use LLM to pick the next tool and refined query based on context. Falls back to heuristic if LLM fails."""

        # Build context from previous searches with sample docs
        history_context = self._build_history_context(previous_results)
        candidate_summary = self._build_candidate_summary(candidate_pool)

        # Build tool descriptions
        tools_desc = "\n".join([f"- {name}: {tool.description}" for name, tool in self.tools.items()])

        # Build machine-readable list of already-tried combinations
        used_pairs_list = [
            {"tool": t, "query": q}
            for (t, q) in used_pairs
        ]
        used_pairs_json = json.dumps(used_pairs_list, ensure_ascii=False)

        prompt = f"""You are a search agent that chooses the best search tool for a query.

Original user query: \"{query}\"

Available tools:
{tools_desc}

{history_context}

{candidate_summary}

Already tried tool+query combinations (DO NOT repeat these):
{used_pairs_json}

Based on the query, current candidates, and search history, decide:
1. Should we continue searching? (yes/no)
2. If yes, which tool should we use next?
3. What specific query should we pass to that tool?

Respond ONLY with valid JSON in this format:
{{
  "continue": true/false,
  "tool": "tool_name" or null,
  "query": "refined query" or null,
  "reasoning": "brief explanation"
}}

Hard constraints:
- You MUST NOT choose any tool+query pair that appears in the already tried list.
- If no new tool+query pair would be helpful, set "continue": false.

Guidelines:
- Use actor_search FIRST when actors are mentioned in the query
- Use genre_search SECOND to filter/refine actor results (e.g., "horror movies with Tom Hanks")
- For queries with both actors AND genres/keywords, search for BOTH to find intersection
- Use semantic/hybrid for general concept searches without specific actors
- Use regex_search for exact phrase matching
- Stop after finding 2-3 complementary searches or when enough results found
- When combining filters (actor + genre), both searches will be intersected automatically

Example strategies:
- "horror movies with Tom Hanks" → actor_search("Tom Hanks") + genre_search("horror")
- "Tom Hanks action movies" → actor_search("Tom Hanks") + genre_search("action")
- "movies about space exploration" → semantic_search or hybrid_search (no actor specified)
"""

        response = execute_llm_prompt(prompt)

        if self.config.debug:
            logger.debug(f"Tool selection raw response: {response}")

        try:
            decision = extract_json_object(response)
            if not isinstance(decision, dict):
                raise ValueError("LLM response is not a JSON object")
            cont = bool(decision.get("continue", False))
            tool_name = decision.get("tool")
            tool_query = decision.get("query")
            reasoning = decision.get("reasoning", "")

            if not cont:
                return None
            if not isinstance(tool_name, str) or not isinstance(tool_query, str):
                logger.debug("LLM returned invalid tool/query types")
                return None

            # If LLM violates the constraint, just stop instead of looping forever
            if (tool_name, tool_query) in used_pairs:
                if self.config.debug:
                    logger.debug(f"LLM suggested repeated tool+query {tool_name}/{tool_query}, stopping.")
                return None

            if tool_name in self.tools:
                return tool_name, tool_query, reasoning

            return None
        except Exception as e:
            logger.warning(f"LLM tool selection failed, using heuristic fallback. Reason: {e}")
            return self._heuristic_tool_choice(query, previous_results, used_pairs)

    def _heuristic_tool_choice(
        self,
        query: str,
        previous_results: list[SearchResult],
        used_pairs: set[tuple[str, str]]
    ) -> tuple[str, str, str] | None:
        """Simple fallback heuristic for tool selection if LLM fails."""
        q_lower = query.lower()
        # Heuristic: actor mention
        if " with " in q_lower or " starring " in q_lower:
            if ("actor_search", query) not in used_pairs:
                return "actor_search", query, "Heuristic: actor mention detected"
        # Default to hybrid_search as a safe bet
        if ("hybrid_search", query) not in used_pairs:
            return "hybrid_search", query, "Heuristic: default hybrid search"
        # If all heuristics exhausted, stop
        return None

    def _merge_results(self, all_results: list[SearchResult], merge_strategy: str = 'auto') -> list[dict[str, Any]]:
        """Merge results from multiple searches with intelligent strategy.

        Args:
            all_results: List of SearchResult objects from different tools
            merge_strategy: 'auto', 'union', or 'intersection'
                - 'auto': Use intersection for refinement searches (actor+genre, etc.)
                - 'union': Combine all unique results (default for similar tools)
                - 'intersection': Only keep movies found by multiple tools

        Returns:
            List of merged and scored movies
        """
        if not all_results:
            if self.config.debug:
                logger.debug("_merge_results: no results to merge")
            return []

        if len(all_results) == 1:
            # Single search - just return its results
            if self.config.debug:
                logger.debug(f"_merge_results: single search, returning {len(all_results[0].results)} results")  # type: ignore[index]
            results = []
            for result in all_results[0].results:  # type: ignore[index]
                result_copy = result.copy()
                result_copy['aggregate_score'] = result.get('score', 1.0)
                result_copy['found_by'] = all_results[0].tool_name  # type: ignore[index]
                results.append(result_copy)
            results.sort(key=lambda x: x.get('aggregate_score', 0), reverse=True)
            return results

        # Determine strategy
        if merge_strategy == 'auto':
            # Check if we have complementary search types that should intersect
            tool_names = [sr.tool_name for sr in all_results]

            # If we have actor + genre/keyword searches, use intersection
            has_actor = 'actor_search' in tool_names
            has_filter = any(t in tool_names for t in ['genre_search', 'keyword_search', 'regex_search'])

            if has_actor and has_filter:
                # Before strict intersection, try filtering actor results by genre-like terms
                all_results = self._refine_actor_results_with_genre(all_results)
                merge_strategy = 'intersection'
                if self.config.debug:
                    logger.debug(f"_merge_results: auto-detected INTERSECTION (actor+filter), tools: {tool_names}")
            else:
                # Similar search types (semantic + keyword + hybrid) -> union
                merge_strategy = 'union'
                if self.config.debug:
                    logger.debug(f"_merge_results: auto-detected UNION, tools: {tool_names}")

        if merge_strategy == 'intersection':
            merged = self._merge_intersection(all_results)
            if self.config.debug:
                logger.debug(f"_merge_intersection returned {len(merged)} results")

            # Fallback: if intersection returns no results, use weighted union instead
            if len(merged) == 0:
                if self.config.debug:
                    logger.warning("Intersection returned 0 results, falling back to weighted union merge")
                logger.info("No movies match ALL criteria. Showing best partial matches instead.")
                merged = self._merge_union(all_results, weighted=True)

            return merged
        else:
            merged = self._merge_union(all_results)
            if self.config.debug:
                logger.debug(f"_merge_union returned {len(merged)} results")
            return merged

    def _merge_union(self, all_results: list[SearchResult], weighted: bool = False) -> list[dict[str, Any]]:
        """Merge results using union - combine all unique results, aggregating scores and found_by across tools."""
        combined: dict[Any, dict[str, Any]] = {}

        for i, search_result in enumerate(all_results):
            base_weight = 1.0 + (i * 0.1)  # Later searches get slight boost

            # Optionally weight certain tools higher when union is fallback
            tool_weight = 1.0
            if weighted:
                if search_result.tool_name == "actor_search":
                    tool_weight = 1.5
                elif search_result.tool_name == "genre_search":
                    tool_weight = 1.2

            weight = base_weight * tool_weight

            for result in search_result.results:
                movie_id = result.get('id')
                if movie_id is None:
                    continue
                score = result.get('score', 1.0) * weight
                if movie_id not in combined:
                    combined[movie_id] = {
                        **result,
                        'aggregate_score': score,
                        'found_by': {search_result.tool_name},
                    }
                else:
                    combined[movie_id]['aggregate_score'] = max(combined[movie_id]['aggregate_score'], score)
                    combined[movie_id]['found_by'].add(search_result.tool_name)

        merged = [
            {**m, 'found_by': ' + '.join(sorted(m['found_by']))}
            for m in combined.values()
        ]
        merged.sort(key=lambda x: x.get('aggregate_score', 0), reverse=True)
        return merged

    def _merge_intersection(self, all_results: list[SearchResult]) -> list[dict[str, Any]]:
        """Merge results using intersection - only keep movies found by multiple tools.

        This is used for refinement queries like "horror movies with Tom Hanks" where
        we want movies that match ALL criteria, not just any criteria.
        """
        if self.config.debug:
            logger.debug(f"_merge_intersection: processing {len(all_results)} search results")
            for sr in all_results:
                logger.debug(f"  - {sr.tool_name}: {len(sr.results)} results")

        # Build a map of movie_id -> list of (tool_name, score, result)
        movie_matches: dict[int, list[tuple[str, float, dict]]] = {}

        for search_result in all_results:
            for result in search_result.results:
                movie_id = result.get('id')
                if movie_id not in movie_matches:
                    movie_matches[movie_id] = []

                movie_matches[movie_id].append((
                    search_result.tool_name,
                    result.get('score', 1.0),
                    result
                ))

        if self.config.debug:
            logger.debug(f"_merge_intersection: found {len(movie_matches)} unique movies")

        # Only keep movies that appear in multiple searches
        # Intersection strategy is now configurable
        mode = getattr(self.config, 'intersection_mode', 'auto')
        min_matches = getattr(self.config, 'min_intersection_matches', 2)
        num_tools = len(all_results)
        if mode == "strict":
            required_matches = num_tools
        elif mode == "loose":
            required_matches = min_matches
        else:  # "auto"
            if num_tools <= 2:
                required_matches = num_tools
            else:
                required_matches = max(min_matches, num_tools - 1)

        if self.config.debug:
            logger.debug(f"_merge_intersection: requiring {required_matches} matches out of {num_tools} searches (mode={mode}, min={min_matches})")

        merged = []
        for movie_id, matches in movie_matches.items():
            if self.config.debug and len(matches) > 1:
                logger.debug(f"  Movie ID {movie_id}: {len(matches)} matches - {[str(m[0]) for m in matches]}")

            if len(matches) >= required_matches:
                # Calculate aggregate score as average of all tool scores
                avg_score = sum(score for _, score, _ in matches) / len(matches)

                # Bonus for matching more tools
                completeness_bonus = 0.1 * (len(matches) - 1)
                aggregate_score = min(1.0, avg_score + completeness_bonus)

                # Take the most complete result dict (prefer first match)
                result_copy = matches[0][2].copy()  # type: ignore[index]
                result_copy['aggregate_score'] = aggregate_score
                result_copy['found_by'] = ' + '.join(set(tool for tool, _, _ in matches))
                result_copy['matched_by_count'] = len(matches)
                result_copy['tool_scores'] = {tool: score for tool, score, _ in matches}

                merged.append(result_copy)

        if self.config.debug:
            logger.debug(f"_merge_intersection: returning {len(merged)} movies that matched {required_matches}+ searches")

        # Sort by aggregate score
        merged.sort(key=lambda x: x.get('aggregate_score', 0), reverse=True)
        return merged

    def _refine_actor_results_with_genre(self, all_results: list[SearchResult]) -> list[SearchResult]:
        """Filter actor_search results using genre-like keywords from genre_search.

        Conservative: only replace actor results if filtering leaves at least
        one result. Otherwise, leave original results to avoid dropping valid
        overlaps in synthetic tests.
        """
        actor_sr = next((sr for sr in all_results if sr.tool_name == 'actor_search'), None)
        genre_sr = next((sr for sr in all_results if sr.tool_name == 'genre_search'), None)
        if not actor_sr or not genre_sr:
            return all_results

        # Derive known genre-ish keywords from GENRE_SYNONYMS
        known_genre_terms = set()
        for canon, syns in GENRE_SYNONYMS.items():
            known_genre_terms.add(canon)
            known_genre_terms.update(syns)

        gq = (genre_sr.query or "").lower()
        # Extract candidate tokens longer than 3 chars that are in known set
        raw_terms = [w.strip() for w in gq.replace(',', ' ').split() if len(w.strip()) > 3]
        genre_terms = {t for t in raw_terms if t in known_genre_terms}

        if not genre_terms:
            return all_results

        def is_genre_like(movie: dict[str, Any]) -> bool:
            text = f"{movie.get('title', '')} {movie.get('description', '')}".lower()
            return any(term in text for term in genre_terms)

        filtered_actor_results = [m for m in actor_sr.results if is_genre_like(m)]

        if self.config.debug:
            logger.debug(
                f"_refine_actor_results_with_genre: filtered actor results from {len(actor_sr.results)} to {len(filtered_actor_results)} using terms {sorted(genre_terms)}"
            )

        # Only replace if we didn't eliminate everything
        if filtered_actor_results:
            new_results = []
            for sr in all_results:
                if sr is actor_sr:
                    new_results.append(SearchResult(
                        tool_name=sr.tool_name,
                        query=sr.query,
                        results=filtered_actor_results,
                        reasoning=sr.reasoning,
                        metadata=sr.metadata,
                    ))
                else:
                    new_results.append(sr)
            return new_results
        return all_results

    def _rerank_with_llm(self, query: str, movies: list[dict[str, Any]]) -> list[dict[str, Any]]:
        """Re-rank candidate movies using the LLM for relevance to the query."""
        if not movies:
            return movies

        # Skip reranking for very few results - not worth the LLM call
        if len(movies) <= 2:
            return movies

        # Build compact payload to stay within context limits
        items = []
        for idx, m in enumerate(movies[:20]):  # cap at 20
            items.append({
                "index": idx,
                "title": m.get("title", ""),
                "description": m.get("description", ""),
                "score": m.get("aggregate_score", m.get("score", 1.0)),
            })

        prompt = f"""
You are re-ranking movie search results for Hoopla.

User query: "{query}"

Below are candidate movies with their current scores.
Re-rank them by how well they match the user's intent.
Respond ONLY with JSON:
{{
  "ranking": [{{"index": int, "relevance": float}}]
}}

Candidates:
{json.dumps(items, ensure_ascii=False, indent=2)}
"""

        try:
            response = execute_llm_prompt(prompt)
            data = extract_json_object(response)
            if not data or "ranking" not in data:
                return movies  # fallback to original order

            ranking = data.get("ranking", [])
            valid_indices = set()
            for r in ranking:
                try:
                    idx = int(r.get("index", -1))
                    rel = float(r.get("relevance", 0.0))
                    rel = max(0.0, min(rel, 1.0))  # Cap relevance between 0.0 and 1.0
                    if isinstance(movies, list) and 0 <= idx < len(movies):
                        movies[idx]["llm_relevance"] = rel
                        valid_indices.add(idx)
                except Exception:
                    continue

            # Require at least 50% of items to be mentioned
            if len(valid_indices) < 0.5 * len(movies):
                if self.config.debug:
                    logger.warning(f"LLM rerank skipped: only {len(valid_indices)} of {len(movies)} items mentioned.")
                return movies

            movies.sort(key=lambda m: m.get("llm_relevance", m.get("aggregate_score", 0)), reverse=True)
            return movies
        except Exception as e:
            if self.config.debug:
                logger.warning(f"LLM rerank failed, using original order: {e}")
            return movies

    def search(self, query: str) -> dict[str, Any]:
        """Execute agentic search with dynamic tool selection."""

        correlation_id = str(uuid.uuid4())
        if self.config.debug:
            logger.debug(f"[CID:{correlation_id}] Starting agentic search for: {query}")

        search_history: list[SearchResult] = []
        candidate_pool: dict[int, dict[str, Any]] = {}
        iteration = 0
        used_pairs: set[tuple[str, str]] = set()  # (tool_name, tool_query)

        while iteration < self.config.max_iterations:
            iteration += 1

            if self.config.debug:
                logger.debug(f"[CID:{correlation_id}] Iteration {iteration}/{self.config.max_iterations}")

            # Early exit heuristic: if we already have actor+genre results, stop
            if len(search_history) >= 2:
                tool_names = {sr.tool_name for sr in search_history}
                if "actor_search" in tool_names and "genre_search" in tool_names:
                    total_results = sum(len(sr.results) for sr in search_history)
                    if total_results >= self.config.final_result_limit:
                        if self.config.debug:
                            logger.debug(f"[CID:{correlation_id}] Already have actor+genre results; stopping early.")
                        break

            # Pick next tool
            tool_decision = self._pick_next_tool(query, search_history, used_pairs, candidate_pool)

            if tool_decision is None:
                if self.config.debug:
                    logger.debug(f"[CID:{correlation_id}] Agent decided to stop searching")
                break

            tool_name, tool_query, reasoning = tool_decision

            # HARD GUARD: don't re-run identical tool+query
            if (tool_name, tool_query) in used_pairs:
                if self.config.debug:
                    logger.debug(f"[CID:{correlation_id}] Skipping repeated tool+query: {tool_name} / {tool_query}")
                break

            used_pairs.add((tool_name, tool_query))

            if self.config.debug:
                logger.debug(f"[CID:{correlation_id}] Selected tool: {tool_name}, query: {tool_query}")
                logger.debug(f"[CID:{correlation_id}] Reasoning: {reasoning}")

            # Execute search
            tool = self.tools[tool_name]
            results = tool.search(tool_query, limit=self.config.max_results_per_tool)
            if results is None:
                results = []

            # Store result
            search_result = SearchResult(
                tool_name=tool_name,
                query=tool_query,
                results=results,
                reasoning=reasoning
            )
            search_history.append(search_result)

            # Update candidate pool with highest scoring version of each movie
            for result in results:
                movie_id = result.get("id")
                if movie_id is None:
                    continue
                existing = candidate_pool.get(movie_id)
                if not existing or result.get("score", 0) > existing.get("score", 0):
                    candidate_pool[movie_id] = result

            if self.config.debug:
                logger.debug(f"[CID:{correlation_id}] Found {len(results)} results; candidate pool size now {len(candidate_pool)}")

        # Merge all results
        merged_results = self._merge_results(search_history)
        # Re-rank with LLM for final ordering
        merged_results = self._rerank_with_llm(query, merged_results)
        # Ensure deterministic final sort by relevance/score
        try:
            merged_results.sort(
                key=lambda m: (
                    1 if "llm_relevance" in m else 0,
                    m.get("llm_relevance", 0.0) if "llm_relevance" in m else m.get("aggregate_score", m.get("score", 0.0)),
                ),
                reverse=True,
            )
        except Exception:
            # Fallback simple sort
            merged_results.sort(key=lambda m: m.get("aggregate_score", m.get("score", 0.0)), reverse=True)

        if self.config.debug:
            contributing_tools = {sr.tool_name for sr in search_history}
            logger.debug(
                f"[CID:{correlation_id}] Final results: %d unique movies, tools used: %s",
                len(merged_results),
                ", ".join(sorted(contributing_tools)),
            )

        return {
            'query': query,
            'iterations': iteration,
            'search_history': search_history,
            'results': merged_results[:self.config.final_result_limit],
            'total_unique_results': len(merged_results)
        }

    def search_and_generate(self, query: str) -> str:
        """Execute search and generate answer with citations."""

        search_output = self.search(query)
        results = search_output['results']

        # Build context for generation (include matched actors/genres when present)
        docs_lines: list[str] = []
        for i, movie in enumerate(results):
            idx = i + 1
            line_parts = [f"[{idx}] {movie.get('title', 'Unknown')} (found by {movie.get('found_by', 'unknown')})"]
            if "actor_query_names" in movie:
                try:
                    line_parts.append(f"    Actor query names: {', '.join(movie['actor_query_names'])}")
                except Exception:
                    pass
            if "matched_genres" in movie:
                try:
                    line_parts.append(f"    Matched genres: {', '.join(movie.get('matched_genres', []))}")
                except Exception:
                    pass
            line_parts.append(f"    Description: {movie.get('description', '')}")
            docs_lines.append("\n".join(line_parts))

        docs = "\n\n".join(docs_lines)

        # Build search history summary
        history_summary = "\n".join([
            f"- {sr.tool_name}: '{sr.query}' → {len(sr.results)} results"
            for sr in search_output['search_history']
        ])

        prompt = f"""Answer the user's question based on movies found through multiple search strategies.

User Query: {query}

Search Strategy Used:
{history_summary}

Found Movies:
{docs}

Instructions:
- Keep the user's query in mind: "{query}"
- Provide a comprehensive answer that addresses the query
- Cite sources using [1], [2], etc. when referencing specific movies
- Mention how the search strategy helped find these results
- Be conversational and helpful
- If the results don't fully answer the question, say so

Answer:"""

        answer = execute_llm_prompt(prompt)

        return answer

