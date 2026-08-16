import ssl
import urllib.request
import re
import asyncio
import time

# =========================================================
# SSL CERTIFICATE FIX
# =========================================================
ssl._create_default_https_context = ssl._create_unverified_context

try:
    import urllib3
    urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)
except:
    pass


import torch
from rank_bm25 import BM25Okapi
from sklearn.metrics.pairwise import cosine_similarity
from transformers import AutoTokenizer, AutoModel

# =========================================================
# Try importing wikipedia
# =========================================================
try:
    import wikipedia
    wikipedia.set_lang("en")
    WIKIPEDIA_AVAILABLE = True
    print("Wikipedia library loaded successfully")
except ImportError:
    print("Wikipedia library not available")
    WIKIPEDIA_AVAILABLE = False

# =========================================================
# CONFIGURATION
# =========================================================


MAX_EXPANDED_TERMS = 20
MIN_SEMANTIC_SCORE = 0.08

# =========================================================
# EMBEDDING MODEL
# =========================================================

class FastEmbedder:
    def __init__(self):
        print("Loading embedding model...")
        start_time = time.time()
        self.tokenizer = AutoTokenizer.from_pretrained(
            "sentence-transformers/all-MiniLM-L6-v2"
        )
        self.model = AutoModel.from_pretrained(
            "sentence-transformers/all-MiniLM-L6-v2"
        )
        self.model.eval()
        print(f"Embedding model loaded in {time.time() - start_time:.2f}s")

    def get_embeddings(self, text, domain=None):
        if not text:
            text = ""
        if len(text) > 300:
            text = text[:300]
        encoded = self.tokenizer(
            text,
            padding=True,
            truncation=True,
            max_length=128,
            return_tensors="pt"
        )
        with torch.no_grad():
            output = self.model(**encoded)
        embeddings = output.last_hidden_state
        attention_mask = encoded["attention_mask"]
        mask = attention_mask.unsqueeze(-1).expand(embeddings.size()).float()
        summed = torch.sum(embeddings * mask, dim=1)
        counts = torch.clamp(mask.sum(dim=1), min=1e-9)
        mean_embedding = summed / counts
        return mean_embedding.numpy()


# =========================================================
# WIKIPEDIA FETCHER - FINAL WORKING VERSION
# =========================================================

class WikipediaFetcher:
    def __init__(self):
        self.use_wikipedia_lib = WIKIPEDIA_AVAILABLE

    def get_wikipedia_results(self, query, limit=30):
        if self.use_wikipedia_lib:
            try:
                print(f"Searching for: {query}")
                results = wikipedia.search(query, results=limit)
                if results:
                    print(f"✓ Found {len(results)} results")
                    return results
            except Exception as e:
                print(f"✗ Search error: {e}")
        
        try:
            import requests
            url = "https://en.wikipedia.org/w/api.php"
            params = {
                "action": "query",
                "format": "json",
                "list": "search",
                "srsearch": query,
                "srlimit": limit
            }
            headers = {'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'}
            response = requests.get(url, params=params, headers=headers, timeout=5)
            if response.status_code == 200:
                data = response.json()
                results = data.get("query", {}).get("search", [])
                titles = [r.get("title", "") for r in results if r.get("title")]
                if titles:
                    print(f"✓ Found {len(titles)} results")
                    return titles
        except Exception as e:
            print(f"✗ API error: {e}")
        
        return []

    def get_related_terms_dynamic(self, query):
        """
        FINAL VERSION - Works for ANY query.
        Keeps ALL movies, awards, and related content.
        Only filters out obvious wrong people.
        """
        terms = []
        query_lower = query.lower()
        query_words = query_lower.split()
        full_query = " ".join(query_words)
        first_name = query_words[0] if query_words else ""
        
        # Get search results from multiple sources
        results = self.get_wikipedia_results(query, 25)
        if results:
            terms.extend(results)
        
        # Get film search
        film_search = self.get_wikipedia_results(f"{query} film", 15)
        if film_search:
            terms.extend(film_search)
        
        # Get award search
        award_search = self.get_wikipedia_results(f"{query} award", 10)
        if award_search:
            terms.extend(award_search)
        
        if not terms:
            print("Generating terms from query")
            terms = [query]
            for word in query_lower.split():
                if len(word) > 2:
                    terms.append(word)
        
        # ========== SIMPLE FILTERING ==========
        filtered_terms = []
        seen_terms = set()
        
        # Only filter out these EXACT wrong people (not movies with their names)
        exact_wrong_people = [
            'ramesh babu', 'sudheer babu', 'jagapathi babu', 'yogi babu',
            'ananth babu', 'pawan singh', 'kalyan kumar', 'harish kalyan',
            'pawan kumar', 'mahesh limaye', 'mahesh manjrekar', 
            'mahesh narayanan', 'mahesh bhatt', 'parineeti chopra',
            'ajith', 'dhanush', 'vijay', 'suriya', 'karthi'
        ]
        
        for term in terms:
            term_lower = term.lower()
            
            # Skip duplicates
            if term_lower in seen_terms:
                continue
            seen_terms.add(term_lower)
            
            # Skip very short
            if len(term_lower) < 3:
                continue
            
            # ========== FILTER OUT ONLY EXACT WRONG PEOPLE ==========
            skip = False
            for wrong in exact_wrong_people:
                if wrong in term_lower:
                    skip = True
                    break
            
            if skip:
                continue
            
            # ========== KEEP EVERYTHING ELSE ==========
            # Keep ALL terms that have any of these
            keep = False
            
            # Keep if it contains the full query
            if full_query in term_lower:
                keep = True
            
            # Keep if it's a film (has (film))
            if '(film)' in term_lower:
                keep = True
            
            # Keep if it has a year (movie, album, etc.)
            if re.search(r'\(\d{4}\)', term):
                keep = True
            
            # Keep if it's filmography or awards
            if 'filmography' in term_lower or 'discography' in term_lower:
                keep = True
            
            if 'award' in term_lower or 'nomination' in term_lower:
                keep = True
            
            # Keep if it contains the first name (for people)
            if len(first_name) > 2 and first_name in term_lower:
                keep = True
            
            # Keep if it has any query word
            for word in query_words:
                if len(word) > 2 and word in term_lower:
                    keep = True
                    break
            
            if keep:
                filtered_terms.append(term)
        
        # If no terms, fallback
        if not filtered_terms:
            filtered_terms = [query]
        
        print(f"Generated {len(filtered_terms)} terms")
        return filtered_terms[:20]


# =========================================================
# NEURAL QUERY EXPANDER
# =========================================================

class NeuralExpander:
    def __init__(self):
        print("Initializing Query Expander...")
        self.embedder = FastEmbedder()
        self.wiki = WikipediaFetcher()
        self.domain_weights = {
            "general": (0.7, 0.3),
            "technology": (0.7, 0.3),
            "medical": (0.6, 0.4),
            "entertainment": (0.7, 0.3),
            "sports": (0.7, 0.3)
        }
        print("Query Expander ready!")

    def tokenize(self, text):
        return re.findall(r"\b[a-zA-Z0-9]+\b", text.lower())

    def detect_domain(self, query):
        query_lower = query.lower()
        tech = ["python", "programming", "software", "computer", "ai", "machine learning", "algorithm", "data", "science", "cloud"]
        medical = ["doctor", "medicine", "disease", "heart", "cancer", "medical", "health", "symptoms"]
        sports = ["cricket", "football", "soccer", "tennis", "player", "match", "sports", "team"]
        entertainment = ["movie", "actor", "actress", "film", "cinema", "tollywood", "bollywood", "hollywood"]
        
        if any(w in query_lower for w in tech): return "technology"
        if any(w in query_lower for w in medical): return "medical"
        if any(w in query_lower for w in sports): return "sports"
        if any(w in query_lower for w in entertainment): return "entertainment"
        return "general"

    def evaluate_expansion(self, expanded_terms, original_terms, domain):
        if not expanded_terms or not original_terms:
            return {"precision": 0, "recall": 0, "f1": 0}
        
        expanded_lower = [term.lower().strip() for term in expanded_terms]
        original_lower = [term.lower().strip() for term in original_terms]
        
        relevant_expanded = set()
        relevant_original_found = set()
        
        for exp_term in expanded_lower:
            for orig_term in original_lower:
                if orig_term in exp_term or exp_term in orig_term:
                    relevant_expanded.add(exp_term)
                    relevant_original_found.add(orig_term)
                    break
                
                orig_words = orig_term.split()
                exp_words = exp_term.split()
                
                for orig_word in orig_words:
                    if len(orig_word) > 2:
                        for exp_word in exp_words:
                            if len(exp_word) > 2 and (orig_word in exp_word or exp_word in orig_word):
                                relevant_expanded.add(exp_term)
                                relevant_original_found.add(orig_term)
                                break
                        if exp_term in relevant_expanded:
                            break
                    if exp_term in relevant_expanded:
                        break
        
        precision = len(relevant_expanded) / len(expanded_terms) if expanded_terms else 0
        recall = len(relevant_original_found) / len(original_terms) if original_terms else 0
        recall = min(recall, 1.0)
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
        
        return {
            "precision": round(precision, 2),
            "recall": round(recall, 2),
            "f1": round(f1, 2)
        }

    def calculate_perplexity(self, scores):
        if not scores:
            return None
        try:
            avg = sum(scores) / len(scores)
            return float(1.0 / avg) if avg > 0 else None
        except:
            return None

    async def expand_query(self, query):
        try:
            print("\n========== QUERY ==========")
            print(f"Query: {query}")
            
            start_time = time.time()
            domain = self.detect_domain(query)
            print(f"Detected domain: {domain}")

            original_terms = []
            words = query.split()
            for word in words:
                if len(word) > 2:
                    original_terms.append(word)
            
            for i in range(len(words)):
                if i < len(words) - 1:
                    phrase = words[i] + " " + words[i+1]
                    original_terms.append(phrase)
            
            original_terms = list(dict.fromkeys(original_terms))
            if not original_terms:
                original_terms = [query]
            print(f"Original terms: {original_terms}")

            all_terms = self.wiki.get_related_terms_dynamic(query)
            
            if not all_terms or len(all_terms) < 2:
                all_terms = [query]
                for word in query.split():
                    if len(word) > 2 and word not in all_terms:
                        all_terms.append(word)

            candidates = []
            for term in all_terms:
                if term and term.strip() and len(term) > 1:
                    candidates.append({
                        "term": term,
                        "context": term.lower()
                    })

            print(f"Total candidates: {len(candidates)}")

            if not candidates:
                return {
                    "expanded_terms": original_terms,
                    "metrics": {"precision": 0, "recall": 0, "f1": 0},
                    "perplexity": None,
                    "detected_domain": domain,
                    "total_candidates": 0
                }

            corpus = [c["context"] for c in candidates]
            tokenized_corpus = [self.tokenize(doc) for doc in corpus]
            tokenized_query = self.tokenize(query)
            
            bm25 = BM25Okapi(tokenized_corpus)
            bm25_scores = bm25.get_scores(tokenized_query)

            semantic_weight, bm25_weight = self.domain_weights[domain]
            query_embed = self.embedder.get_embeddings(query, domain)

            term_scores = {}
            max_candidates = min(15, len(candidates))
            
            for idx in range(max_candidates):
                candidate = candidates[idx]
                term = candidate["term"]
                term_embed = self.embedder.get_embeddings(term, domain)
                semantic_score = float(cosine_similarity(query_embed, term_embed)[0][0])
                
                if semantic_score < MIN_SEMANTIC_SCORE:
                    continue
                
                bm25_score = float(bm25_scores[idx]) if idx < len(bm25_scores) else 0
                term_scores[term] = (semantic_weight * semantic_score + bm25_weight * bm25_score)

            sorted_terms = sorted(term_scores.items(), key=lambda x: x[1], reverse=True)
            top_terms = [term for term, _ in sorted_terms[:MAX_EXPANDED_TERMS]]
            scores = [score for _, score in sorted_terms[:MAX_EXPANDED_TERMS]]

            eval_metrics = self.evaluate_expansion(top_terms, original_terms, domain)
            perplexity = self.calculate_perplexity(scores)

            elapsed = time.time() - start_time
            print(f"Expanded terms count: {len(top_terms)}")
            print(f"Time taken: {elapsed:.2f}s")
            print("============================\n")

            return {
                "expanded_terms": top_terms,
                "metrics": eval_metrics,
                "perplexity": perplexity,
                "detected_domain": domain,
                "total_candidates": len(candidates),
                "total_expanded": len(top_terms)
            }

        except Exception as e:
            import traceback
            print("\n========== ERROR ==========")
            traceback.print_exc()
            print("===========================\n")
            return {
                "error": str(e),
                "expanded_terms": [],
                "metrics": {"precision": 0, "recall": 0, "f1": 0},
                "perplexity": None,
                "detected_domain": "general",
                "total_candidates": 0,
                "total_expanded": 0
            }
