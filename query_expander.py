import asyncio
import aiohttp
import wikipedia
import spacy
import warnings
from transformers import AutoTokenizer, AutoModel
from sklearn.metrics.pairwise import cosine_similarity
from collections import defaultdict
import math
import nest_asyncio
from rank_bm25 import BM25Okapi
from nltk.stem import PorterStemmer
import numpy as np

warnings.filterwarnings("ignore")
nest_asyncio.apply()

# Constants
MAX_CANDIDATES = 7
WIKIDATA_LIMIT = 25
MIN_TERMS_FOR_METRICS = 3
MIN_SEMANTIC_SCORE = 0.35
MEDICAL_BLACKLIST = {"institute", "society", "journal", "college", "university", "hospital", "peshawar", "rawalpindi"}

nlp = spacy.load("en_core_web_sm")
stemmer = PorterStemmer()

class HybridEmbedder:
    def __init__(self):
        self.general_tokenizer = AutoTokenizer.from_pretrained("sentence-transformers/all-MiniLM-L6-v2")
        self.general_model = AutoModel.from_pretrained("sentence-transformers/all-MiniLM-L6-v2")
        self.domain_tokenizer = AutoTokenizer.from_pretrained("sentence-transformers/paraphrase-multilingual-mpnet-base-v2")
        self.domain_model = AutoModel.from_pretrained("sentence-transformers/paraphrase-multilingual-mpnet-base-v2")

    def get_embeddings(self, text, domain):
        if domain == "anime":
            inputs = self.domain_tokenizer(text, return_tensors="pt", truncation=True, max_length=256)
            outputs = self.domain_model(**inputs)
        else:
            inputs = self.general_tokenizer(text, return_tensors="pt", truncation=True, max_length=256)
            outputs = self.general_model(**inputs)
        return outputs.last_hidden_state.mean(dim=1).detach().numpy()

class NeuralExpander:
    def __init__(self):
        self.session = None
        self.entity_cache = defaultdict(dict)
        self.embedder = HybridEmbedder()
        self.domain_weights = {
            "medical": (0.8, 0.2),
            "anime": (0.7, 0.3),
            "science": (0.6, 0.4),
            "tech": (0.5, 0.5),
            "general": (0.6, 0.4)
        }
        self.domain_properties = {
            "medical": ['P1995', 'P279', 'P1057'],  # Medical specialty, subclass of, medical condition
            "general": ['P31', 'P361'],
            "anime": ['P527', 'P674'],
            "science": ['P361', 'P921'],
            "tech": ['P178', 'P400']
        }

    def detect_domain(self, query):
        doc = nlp(query.lower())
        medical_keywords = {"cardiology", "medical", "surgery", "anatomy", "physiology"}
        domain_keywords = {
            "anime": {"anime", "manga", "otaku"},
            "science": {"quantum", "physics", "biology"},
            "tech": {"python", "javascript", "ai"}
        }
        
        if any(token.text in medical_keywords for token in doc):
            return "medical"
        for domain, keywords in domain_keywords.items():
            if any(token.text in keywords for token in doc):
                return domain
        return "general"

    async def init_session(self):
        if self.session is None or self.session.closed:
            self.session = aiohttp.ClientSession()

    async def close(self):
        if self.session:
            await self.session.close()

    async def fetch_wikidata(self, entity_id):
        try:
            async with self.session.get(
                "https://www.wikidata.org/w/api.php",
                params={
                    "action": "wbgetentities",
                    "ids": entity_id,
                    "props": "labels|aliases|claims|descriptions",
                    "languages": "en",
                    "format": "json"
                },
                timeout=3
            ) as response:
                return await response.json()
        except:
            return None

    async def get_ml_terms(self, entity_id, domain):
        try:
            data = await self.fetch_wikidata(entity_id)
            if not data or 'entities' not in data:
                return []

            entity = data['entities'][entity_id]
            terms = []
            terms.append(entity['labels']['en']['value'])
            terms.extend([alias['value'] for alias in entity['aliases'].get('en', [])][:5])

            for prop in self.domain_properties.get(domain, self.domain_properties["general"]):
                if prop in entity.get('claims', {}):
                    for claim in entity['claims'][prop][:5]:
                        try:
                            target_id = claim['mainsnak']['datavalue']['value']['id']
                            target_data = await self.fetch_wikidata(target_id)
                            if target_data:
                                terms.append(target_data['entities'][target_id]['labels']['en']['value'])
                        except:
                            continue
            return list(set(terms))[:WIKIDATA_LIMIT]
        except:
            return []

    async def get_wikipedia_page(self, title):
        try:
            return wikipedia.page(title, auto_suggest=True, preload=False)
        except:
            return None

    async def disambiguate_term(self, query, term, domain):
        try:
            search_query = f"{term} {query}" if term != query else term
            search_results = wikipedia.search(search_query, results=MAX_CANDIDATES)

            best_score = -1
            best_entity = {"title": term, "qid": None, "content": ""}

            for candidate in search_results:
                try:
                    page = await self.get_wikipedia_page(candidate)
                    if not page or not hasattr(page, 'summary') or not page.summary:
                        continue

                    qid = getattr(page, 'pageprops', {}).get('wikibase_item', None)
                    content = f"{page.title} {page.summary[:300]}"

                    query_embed = self.embedder.get_embeddings(query, domain)
                    candidate_embed = self.embedder.get_embeddings(content, domain)
                    similarity = cosine_similarity(query_embed, candidate_embed)[0][0]

                    if similarity > best_score:
                        best_score = similarity
                        best_entity = {
                            "title": page.title,
                            "qid": qid,
                            "content": content.lower()
                        }
                except:
                    continue
            return best_entity
        except:
            return {"title": term, "qid": None, "content": ""}

    def tokenize(self, text):
        doc = nlp(text.lower())
        return [stemmer.stem(token.lemma_) for token in doc if token.is_alpha]

    def evaluate_expansion(self, predicted_terms, original_terms, domain):
        if len(predicted_terms) < MIN_TERMS_FOR_METRICS or len(original_terms) == 0:
            return {
                "precision": 0.0,
                "recall": 0.0,
                "f1": 0.0
            }

        # Get core stems from original query
        core_stems = set()
        for term in original_terms:
            for token in term.lower().split():
                core_stems.add(stemmer.stem(token))

        # Calculate coverage
        covered_stems = set()
        relevant_terms = 0
        
        for p_term in predicted_terms:
            term_stems = set()
            for token in p_term.lower().split():
                term_stems.add(stemmer.stem(token))
            
            # Check relevance and update coverage
            if not term_stems.isdisjoint(core_stems):
                relevant_terms += 1
                covered_stems.update(term_stems & core_stems)

        # Calculate metrics
        precision = relevant_terms / len(predicted_terms) if predicted_terms else 0.0
        recall = len(covered_stems) / len(core_stems) if core_stems else 0.0
        f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0

        return {
            "precision": round(float(precision), 2),
            "recall": round(float(recall), 2),
            "f1": round(float(f1), 2)
        }

    def calculate_perplexity(self, scores):
        if len(scores) < 2:
            return float('inf')
        
        max_score = max(scores)
        exp_scores = [math.exp(s - max_score) for s in scores]
        sum_exp = sum(exp_scores)
        probabilities = [es / sum_exp for es in exp_scores]
        
        entropy = -sum(p * math.log(p + 1e-8) for p in probabilities)
        return round(float(math.exp(entropy)), 2)

    async def expand_query(self, query):
        await self.init_session()
        try:
            domain = self.detect_domain(query)
            doc = nlp(query)
            original_terms = list(set(
                [chunk.text for chunk in doc.noun_chunks] +
                [ent.text for ent in doc.ents]
            ))

            candidates = []
            entities = await asyncio.gather(*[
                self.disambiguate_term(query, term, domain) 
                for term in original_terms
            ])

            for entity in entities:
                if entity["title"]:
                    candidates.append({
                        "term": entity["title"],
                        "context": entity["content"],
                        "source": "wikipedia"
                    })

            wiki_tasks = [
                self.get_ml_terms(entity["qid"], domain) 
                for entity in entities if entity["qid"]
            ]
            wiki_results = await asyncio.gather(*wiki_tasks)
            
            for term_list in wiki_results:
                candidates.extend({
                    "term": term,
                    "context": term.lower(),
                    "source": "wikidata"
                } for term in term_list)

            # Medical domain filtering
            if domain == "medical":
                candidates = [c for c in candidates if not any(
                    kw in c["term"].lower() for kw in MEDICAL_BLACKLIST
                )]

            if len(candidates) < 5:
                fallback_terms = wikipedia.search(query, results=15)
                for term in fallback_terms:
                    if term.lower() not in {c["term"].lower() for c in candidates}:
                        candidates.append({
                            "term": term,
                            "context": term.lower(),
                            "source": "fallback"
                        })

            corpus = [c["context"] for c in candidates]
            tokenized_corpus = [self.tokenize(doc) for doc in corpus]
            bm25 = BM25Okapi(tokenized_corpus)
            tokenized_query = self.tokenize(query)

            semantic_weight, bm25_weight = self.domain_weights[domain]
            query_embed = self.embedder.get_embeddings(query, domain)
            
            term_scores = {}
            for idx, candidate in enumerate(candidates):
                term = candidate["term"]
                term_embed = self.embedder.get_embeddings(term, domain)
                semantic_score = float(cosine_similarity(query_embed, term_embed)[0][0])
                
                if semantic_score < MIN_SEMANTIC_SCORE:
                    continue
                
                bm25_score = float(bm25.get_scores(tokenized_query)[idx])
                
                term_scores[term] = (
                    semantic_weight * semantic_score +
                    bm25_weight * bm25_score
                )

            sorted_terms = sorted(term_scores.items(), key=lambda x: x[1], reverse=True)
            top_terms = [term for term, _ in sorted_terms[:15]]
            scores = [score for _, score in sorted_terms[:15]]

            eval_metrics = self.evaluate_expansion(top_terms, original_terms, domain)
            perplexity = self.calculate_perplexity(scores)

            return {
                "expanded_terms": top_terms,
                "metrics": eval_metrics,
                "perplexity": perplexity,
                "detected_domain": domain
            }

        except Exception as e:
            print(f"Error: {str(e)}")
            return {"error": str(e)}