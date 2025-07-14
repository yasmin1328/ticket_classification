import openai
import json
import time
from typing import List, Dict, Optional, Tuple
import logging
from collections import Counter
from config import Config

class EnhancedLLMServiceWithConfidence:
    def __init__(self, api_key: str = None, model: str = None):
        self.logger = logging.getLogger(__name__)
        self.api_key = api_key or Config.OPENAI_API_KEY
        self.model = model or Config.OPENAI_MODEL
        
        if self.api_key:
            openai.api_key = self.api_key
    
    def classify_incident(self, 
                         description: str, 
                         similar_examples: List[Dict], 
                         language: str = 'ar') -> Dict:
        """Classify an incident with enhanced confidence calculation"""
        
        # First, try to classify using similarity-based approach
        similarity_result = self._classify_using_similarity_enhanced(description, similar_examples)
        
        # If we have good similarity results, use them
        if similarity_result['confidence'] >= 0.5 or len(similar_examples) >= 3:
            return similarity_result
        
        # Otherwise, try LLM enhancement if API is available
        if self.api_key:
            try:
                llm_result = self._classify_with_llm(description, similar_examples, language)
                
                # Combine similarity and LLM results for better confidence
                combined_result = self._combine_similarity_and_llm_results(
                    similarity_result, llm_result, description
                )
                return combined_result
                    
            except Exception as e:
                self.logger.warning(f"LLM classification failed, using enhanced similarity: {e}")
                return similarity_result
        
        return similarity_result
    
    def _classify_using_similarity_enhanced(self, description: str, similar_examples: List[Dict]) -> Dict:
        """Enhanced similarity-based classification with improved confidence calculation"""
        
        if not similar_examples:
            return self._create_new_category_fallback(description)
        
        # Extract categories from similar examples with their similarity scores as weights
        category_votes = {}
        total_weight = 0
        
        for example in similar_examples:
            if 'metadata' not in example or not example['metadata']:
                continue
                
            metadata = example['metadata']
            similarity_score = example.get('similarity_score', 0)
            
            # Weight votes by similarity score
            weight = similarity_score
            total_weight += weight
            
            cat1 = metadata.get('category1', '').strip()
            cat2 = metadata.get('category2', '').strip()
            
            if cat1 and cat2:
                category_key = f"{cat1}|{cat2}"
                if category_key not in category_votes:
                    category_votes[category_key] = {
                        'weight': 0,
                        'count': 0,
                        'cat1': cat1,
                        'cat2': cat2,
                        'examples': []
                    }
                
                category_votes[category_key]['weight'] += weight
                category_votes[category_key]['count'] += 1
                category_votes[category_key]['examples'].append({
                    'description': metadata.get('description', ''),
                    'similarity': similarity_score
                })
        
        if not category_votes:
            return self._create_new_category_fallback(description)
        
        # Find the category with highest weighted vote
        best_category = max(category_votes.items(), key=lambda x: x[1]['weight'])
        category_data = best_category[1]
        
        # ENHANCED CONFIDENCE CALCULATION
        enhanced_confidence = self._calculate_enhanced_confidence(
            description, 
            category_data,
            similar_examples,
            total_weight
        )
        
        # Generate reasoning
        reasoning = self._generate_enhanced_reasoning(
            category_data['cat1'], 
            category_data['cat2'], 
            category_data['examples'],
            enhanced_confidence
        )
        
        return {
            'subdirectory1': category_data['cat1'],
            'subdirectory2': category_data['cat2'],
            'confidence': round(enhanced_confidence, 3),
            'reasoning': reasoning,
            'domain_relevance': self._calculate_domain_relevance(description, category_data),
            'category_source': 'similarity_voting_enhanced',
            'supporting_examples': len(category_data['examples']),
            'vote_weight': round(category_data['weight'], 3),
            'enhancement_details': self._get_confidence_breakdown(description, category_data, similar_examples)
        }
    
    def _calculate_enhanced_confidence(self, description: str, category_data: Dict, 
                                     all_examples: List[Dict], total_weight: float) -> float:
        """
        Enhanced confidence calculation that addresses low confidence issues
        while maintaining classification accuracy
        """
        
        # Extract key metrics
        max_similarity = max(ex['similarity'] for ex in category_data['examples'])
        avg_similarity = sum(ex['similarity'] for ex in category_data['examples']) / len(category_data['examples'])
        supporting_examples = len(category_data['examples'])
        vote_weight = category_data['weight']
        
        # 1. Base Similarity Confidence (40% weight)
        similarity_confidence = max_similarity * 0.6 + avg_similarity * 0.4
        
        # 2. Support Ratio (30% weight) - How many examples support this category
        total_examples = len(all_examples)
        support_ratio = min(supporting_examples / min(total_examples, 10), 1.0) if total_examples > 0 else 0
        
        # 3. Vote Strength (20% weight) - Relative voting power
        vote_strength = vote_weight / total_weight if total_weight > 0 else 0
        
        # 4. Quality Bonuses (up to 15% boost)
        quality_bonus = 0
        
        # High similarity bonus
        if max_similarity >= 0.85:
            quality_bonus += 0.15
        elif max_similarity >= 0.75:
            quality_bonus += 0.12
        elif max_similarity >= 0.65:
            quality_bonus += 0.08
        elif max_similarity >= 0.55:
            quality_bonus += 0.05
        
        # Multiple examples bonus
        if supporting_examples >= 5:
            quality_bonus += 0.05
        elif supporting_examples >= 3:
            quality_bonus += 0.03
        
        # 5. Relevance Bonus (up to 10% boost)
        relevance_bonus = self._calculate_relevance_bonus(
            description, 
            category_data['cat1'], 
            category_data['cat2']
        )
        
        # 6. Consistency Bonus (up to 5% boost)
        consistency_bonus = self._calculate_consistency_bonus(category_data['examples'])
        
        # Calculate enhanced confidence
        enhanced_confidence = (
            similarity_confidence * 0.40 +     # Base similarity (40%)
            support_ratio * 0.30 +             # Support ratio (30%)
            vote_strength * 0.20 +             # Vote strength (20%)
            quality_bonus +                    # Quality bonuses (up to 15%)
            relevance_bonus +                  # Relevance bonus (up to 10%)
            consistency_bonus                  # Consistency bonus (up to 5%)
        )
        
        # Apply intelligent floors for clearly good classifications
        if max_similarity >= 0.7 and supporting_examples >= 3 and relevance_bonus > 0.05:
            enhanced_confidence = max(enhanced_confidence, 0.70)  # Strong evidence floor
        elif max_similarity >= 0.6 and supporting_examples >= 2 and relevance_bonus > 0.03:
            enhanced_confidence = max(enhanced_confidence, 0.60)  # Good evidence floor
        elif max_similarity >= 0.5 and supporting_examples >= 1:
            enhanced_confidence = max(enhanced_confidence, 0.45)  # Minimum evidence floor
        
        # Apply intelligent ceiling
        enhanced_confidence = min(enhanced_confidence, 0.95)  # Leave room for uncertainty
        
        return enhanced_confidence
    
    def _calculate_relevance_bonus(self, query_text: str, category1: str, category2: str) -> float:
        """Calculate bonus based on logical relevance of classification"""
        
        query_lower = query_text.lower()
        cat1_lower = category1.lower()
        cat2_lower = category2.lower()
        
        # Enhanced keyword mappings with more comprehensive coverage
        relevance_mappings = {
            'payment_high': {
                'keywords': ['دفع', 'فاتورة', 'سداد', 'مدفوعات', 'تسديد', 'دفعة', 'حساب',
                           'payment', 'invoice', 'pay', 'bill', 'charge', 'fee'],
                'categories': ['مدفوعات', 'فاتورة', 'سداد', 'دفع', 'حساب',
                             'payment', 'invoice', 'billing', 'fee'],
                'bonus': 0.10
            },
            'login_high': {
                'keywords': ['دخول', 'تسجيل', 'مصادقة', 'كلمة مرور', 'باسورد', 'حساب', 'هوية',
                           'login', 'password', 'authentication', 'access', 'signin', 'account'],
                'categories': ['دخول', 'تسجيل', 'مصادقة', 'هوية', 'حساب',
                             'login', 'authentication', 'access', 'account'],
                'bonus': 0.10
            },
            'certificate_high': {
                'keywords': ['شهادة', 'وثيقة', 'مستند', 'تصريح', 'إثبات', 'إرسالية',
                           'certificate', 'document', 'credential', 'proof'],
                'categories': ['شهادة', 'وثائق', 'مستندات', 'إرسالية', 'شهادات',
                             'certificate', 'document', 'credential'],
                'bonus': 0.08
            },
            'system_medium': {
                'keywords': ['نظام', 'خطأ', 'عطل', 'تقني', 'تعطل', 'مشكلة', 'تحديث',
                           'system', 'error', 'technical', 'issue', 'problem', 'bug'],
                'categories': ['نظام', 'تقني', 'خطأ', 'عطل', 'مشاكل',
                             'system', 'technical', 'error', 'issue'],
                'bonus': 0.06
            },
            'network_medium': {
                'keywords': ['شبكة', 'انترنت', 'اتصال', 'موقع', 'خادم', 'سيرفر',
                           'network', 'internet', 'connection', 'server', 'website'],
                'categories': ['شبكة', 'اتصال', 'انترنت', 'خادم',
                             'network', 'connection', 'internet', 'server'],
                'bonus': 0.06
            }
        }
        
        # Check for high relevance matches
        for mapping_type, mapping in relevance_mappings.items():
            query_has_keywords = any(keyword in query_lower for keyword in mapping['keywords'])
            category_matches = any(
                cat in cat1_lower or cat in cat2_lower 
                for cat in mapping['categories']
            )
            
            if query_has_keywords and category_matches:
                return mapping['bonus']
        
        # Check for partial matches (lower bonus)
        all_keywords = []
        for mapping in relevance_mappings.values():
            all_keywords.extend(mapping['keywords'])
        
        if any(keyword in query_lower for keyword in all_keywords):
            return 0.03  # Partial relevance bonus
        
        return 0.0
    
    def _calculate_consistency_bonus(self, examples: List[Dict]) -> float:
        """Calculate bonus based on consistency of similar examples"""
        
        if len(examples) < 2:
            return 0.0
        
        similarities = [ex['similarity'] for ex in examples]
        
        # High consistency: all examples have similar high scores
        min_sim = min(similarities)
        max_sim = max(similarities)
        
        # Consistency measure: smaller spread = more consistent
        consistency = 1 - (max_sim - min_sim)
        
        # Bonus for high consistency
        if consistency >= 0.9 and min_sim >= 0.6:
            return 0.05
        elif consistency >= 0.8 and min_sim >= 0.5:
            return 0.03
        elif consistency >= 0.7:
            return 0.01
        
        return 0.0
    
    def _calculate_domain_relevance(self, description: str, category_data: Dict) -> float:
        """Calculate domain relevance score"""
        
        max_similarity = max(ex['similarity'] for ex in category_data['examples'])
        
        # Domain relevance based on similarity and category confidence
        relevance_bonus = self._calculate_relevance_bonus(
            description, 
            category_data['cat1'], 
            category_data['cat2']
        )
        
        base_relevance = min(max_similarity + 0.1, 1.0)
        enhanced_relevance = min(base_relevance + relevance_bonus, 1.0)
        
        return enhanced_relevance
    
    def _generate_enhanced_reasoning(self, cat1: str, cat2: str, examples: List[Dict], confidence: float) -> str:
        """Generate enhanced reasoning with confidence explanation"""
        
        example_count = len(examples)
        avg_similarity = sum(ex['similarity'] for ex in examples) / example_count
        max_similarity = max(ex['similarity'] for ex in examples)
        
        reasoning = f"تم تصنيف هذه الحادثة إلى '{cat1} -> {cat2}' بناءً على تحليل {example_count} حادثة مشابهة. "
        reasoning += f"متوسط درجة التشابه: {avg_similarity:.3f}، أعلى درجة تشابه: {max_similarity:.3f}. "
        
        # Enhanced confidence explanation
        if confidence >= 0.8:
            reasoning += "درجة الثقة عالية جداً نظراً للتطابق القوي مع الحوادث الموجودة والتصنيف المنطقي للمشكلة."
        elif confidence >= 0.7:
            reasoning += "درجة الثقة عالية تشير إلى تطابق واضح مع الفئة المحددة."
        elif confidence >= 0.6:
            reasoning += "درجة الثقة جيدة مما يدل على تطابق واضح مع الفئة المحددة."
        elif confidence >= 0.5:
            reasoning += "درجة الثقة متوسطة، التصنيف محتمل ومدعوم بأمثلة مشابهة."
        else:
            reasoning += "درجة الثقة منخفضة، قد يحتاج التصنيف إلى مراجعة يدوية."
        
        return reasoning
    
    def _get_confidence_breakdown(self, description: str, category_data: Dict, all_examples: List[Dict]) -> Dict:
        """Get detailed breakdown of confidence calculation for debugging"""
        
        max_similarity = max(ex['similarity'] for ex in category_data['examples'])
        avg_similarity = sum(ex['similarity'] for ex in category_data['examples']) / len(category_data['examples'])
        supporting_examples = len(category_data['examples'])
        
        breakdown = {
            'max_similarity': max_similarity,
            'avg_similarity': avg_similarity,
            'supporting_examples': supporting_examples,
            'total_examples': len(all_examples),
            'support_ratio': min(supporting_examples / min(len(all_examples), 10), 1.0),
            'relevance_bonus': self._calculate_relevance_bonus(description, category_data['cat1'], category_data['cat2']),
            'consistency_bonus': self._calculate_consistency_bonus(category_data['examples']),
            'quality_bonus': 0.15 if max_similarity >= 0.85 else (0.12 if max_similarity >= 0.75 else 0.08)
        }
        
        return breakdown
    
    def _combine_similarity_and_llm_results(self, similarity_result: Dict, llm_result: Dict, description: str) -> Dict:
        """Combine similarity and LLM results for optimal confidence"""
        
        # Use similarity result as base, enhance with LLM insights
        if llm_result.get('confidence', 0) > similarity_result.get('confidence', 0):
            # LLM has higher confidence, but keep similarity metadata
            combined = llm_result.copy()
            combined['category_source'] = 'llm_enhanced_similarity'
            combined['similarity_metadata'] = similarity_result.get('enhancement_details', {})
        else:
            # Similarity has higher confidence
            combined = similarity_result.copy()
            combined['llm_validation'] = llm_result.get('reasoning', '')
        
        return combined
    
    def _classify_with_llm(self, description: str, similar_examples: List[Dict], language: str) -> Dict:
        """Enhanced LLM classification using similar examples"""
        
        prompt = self._build_enhanced_classification_prompt(description, similar_examples, language)
        
        try:
            response = self._call_llm(prompt, max_tokens=500, temperature=0.2)
            result = self._parse_classification_response(response, description)
            
            # Enhance with similarity context
            if similar_examples:
                max_sim = max(ex.get('similarity_score', 0) for ex in similar_examples)
                result['confidence'] = min(result.get('confidence', 0.5) + (max_sim * 0.2), 1.0)
                result['similarity_context'] = {
                    'similar_incidents_count': len(similar_examples),
                    'max_similarity': max_sim,
                    'category_source': 'llm_enhanced'
                }
            
            return result
            
        except Exception as e:
            self.logger.error(f"LLM classification failed: {e}")
            raise
    
    def _build_enhanced_classification_prompt(self, description: str, similar_examples: List[Dict], language: str) -> str:
        """Build enhanced prompt with better similar examples presentation"""
        
        # Prepare similar examples with clear categorization
        examples_analysis = ""
        if similar_examples:
            examples_analysis = "\n\nالحوادث المشابهة الموجودة في قاعدة البيانات:\n"
            
            for i, example in enumerate(similar_examples[:3], 1):  # Top 3 most similar
                meta = example.get('metadata', {})
                similarity = example.get('similarity_score', 0)
                
                examples_analysis += f"\n{i}. درجة التشابه: {similarity:.3f}\n"
                examples_analysis += f"   الوصف: {meta.get('description', 'غير متوفر')}\n"
                examples_analysis += f"   ✅ الفئة الرئيسية: {meta.get('category1', 'غير محدد')}\n"
                examples_analysis += f"   ✅ الفئة الفرعية: {meta.get('category2', 'غير محدد')}\n"
            
            # Category frequency analysis
            categories = {}
            for ex in similar_examples:
                meta = ex.get('metadata', {})
                cat_key = f"{meta.get('category1', '')} -> {meta.get('category2', '')}"
                categories[cat_key] = categories.get(cat_key, 0) + 1
            
            examples_analysis += f"\n📊 توزيع الفئات في النتائج المشابهة:\n"
            for cat, count in sorted(categories.items(), key=lambda x: x[1], reverse=True):
                examples_analysis += f"   • {cat}: {count} حادثة\n"
        
        prompt = f"""أنت نظام ذكي متقدم لتصنيف الحوادث التقنية في شركة. مهمتك تصنيف الحادثة بناءً على الوصف والحوادث المشابهة.

🎯 الحادثة المطلوب تصنيفها:
"{description}"

{examples_analysis}

📋 مهمتك:
1. حلل الوصف بعناية
2. استفد من الحوادث المشابهة لتحديد التصنيف الأنسب
3. اختر الفئة الرئيسية والفرعية الأكثر دقة
4. قدم مستوى ثقة عالي (0.7-1.0) إذا كان التصنيف واضح

⚠️ قواعد مهمة:
- استخدم نفس أسماء الفئات الموجودة في الأمثلة المشابهة
- إذا كانت الحوادث المشابهة متفقة على فئة معينة، اختر نفس الفئة
- مستوى الثقة يجب أن يعكس وضوح التصنيف ودرجة التشابه

قم بالرد في شكل JSON فقط:
{{
    "subdirectory1": "الفئة الرئيسية",
    "subdirectory2": "الفئة الفرعية", 
    "confidence": 0.85,
    "reasoning": "تبرير مفصل للتصنيف",
    "domain_relevance": 0.9
}}"""

        return prompt
    
    def _create_new_category_fallback(self, description: str) -> Dict:
        """Create meaningful fallback when no similar examples available"""
        
        # Simple rule-based categorization based on keywords
        desc_lower = description.lower()
        
        # Enhanced keyword mappings from config
        category_keywords = getattr(Config, 'CATEGORY_KEYWORDS', {})
        
        # Find matching category
        for (cat1, cat2), keywords in category_keywords.items():
            if any(keyword in desc_lower for keyword in keywords):
                confidence = 0.55  # Higher confidence for intelligent rule-based
                reasoning = f'تم تصنيف هذه الحادثة إلى "{cat1} -> {cat2}" بناءً على تحليل الكلمات المفتاحية في الوصف. درجة الثقة متوسطة ويُنصح بالمراجعة.'
                
                return {
                    'subdirectory1': cat1,
                    'subdirectory2': cat2,
                    'confidence': confidence,
                    'reasoning': reasoning,
                    'domain_relevance': 0.7,
                    'category_source': 'rule_based_intelligent',
                    'enhancement_details': {'rule_matched': f"{cat1}|{cat2}"}
                }
        
        # Ultimate fallback - but more specific
        return {
            'subdirectory1': 'طلبات المساعدة العامة',
            'subdirectory2': 'طلب مساعدة غير مصنف',
            'confidence': 0.35,
            'reasoning': 'لم يتم العثور على حوادث مشابهة في قاعدة البيانات. تم تصنيف الحادثة كطلب مساعدة عام يحتاج مراجعة يدوية.',
            'domain_relevance': 0.5,
            'category_source': 'fallback_smart'
        }
    
    def _call_llm(self, prompt: str, max_tokens: int = 500, temperature: float = 0.3) -> str:
        """Make API call to OpenAI with better error handling"""
        
        if not self.api_key:
            raise ValueError("No API key configured")

        try:
            response = openai.ChatCompletion.create(
                model=self.model,
                messages=[
                    {
                        "role": "system", 
                        "content": "أنت نظام ذكي متخصص في تصنيف الحوادث التقنية بدقة عالية. تجيب باللغة العربية وتستخدم الفئات الموجودة في قاعدة البيانات."
                    },
                    {"role": "user", "content": prompt}
                ],
                max_tokens=max_tokens,
                temperature=temperature,
                top_p=0.9
            )

            return response['choices'][0]['message']['content'].strip()

        except Exception as e:
            self.logger.error(f"LLM API call failed: {str(e)}")
            raise
    
    def _parse_classification_response(self, response: str, original_description: str) -> Dict:
        """Parse and validate LLM classification response"""
        try:
            # Try to extract JSON from response
            json_start = response.find('{')
            json_end = response.rfind('}') + 1
            
            if json_start != -1 and json_end > json_start:
                json_str = response[json_start:json_end]
                result = json.loads(json_str)
            else:
                result = json.loads(response)
            
            # Validate and set defaults
            result.setdefault('subdirectory1', 'غير محدد')
            result.setdefault('subdirectory2', 'غير محدد')
            result.setdefault('confidence', 0.5)
            result.setdefault('reasoning', 'تم التصنيف باستخدام الذكاء الاصطناعي')
            result.setdefault('domain_relevance', 0.7)
            result.setdefault('category_source', 'llm')
            
            # Ensure confidence is between 0 and 1
            result['confidence'] = max(0.0, min(1.0, float(result['confidence'])))
            
            return result
            
        except Exception as e:
            self.logger.error(f"Error parsing LLM response: {str(e)}")
            raise
    
    def create_new_category(self, description: str, language: str = 'ar') -> Dict:
        """Enhanced new category creation"""
        
        if self.api_key:
            try:
                prompt = self._build_new_category_prompt(description, language)
                response = self._call_llm(prompt, max_tokens=300, temperature=0.5)
                result = self._parse_classification_response(response, description)
                result['category_source'] = 'llm_created'
                return result
            except Exception as e:
                self.logger.warning(f"LLM new category creation failed: {e}")
        
        # Fallback to intelligent rule-based creation
        return self._create_new_category_fallback(description)
    
    def _build_new_category_prompt(self, description: str, language: str) -> str:
        """Build prompt for creating new category"""
        
        prompt = f"""أنت نظام ذكي لإنشاء فئات جديدة للحوادث التقنية.

الوصف الجديد: "{description}"

هذا الوصف لا يتطابق بدرجة كافية مع الفئات الموجودة في قاعدة البيانات.

مهمتك إنشاء فئة رئيسية وفرعية مناسبة:

قم بالرد في شكل JSON:
{{
    "subdirectory1": "الفئة الرئيسية الجديدة",
    "subdirectory2": "الفئة الفرعية الجديدة",
    "confidence": 0.75,
    "reasoning": "تبرير إنشاء هذه الفئة الجديدة",
    "domain_relevance": 0.8
}}"""

        return prompt
    
    def summarize_incident(self, description: str, language: str = 'ar') -> str:
        """Enhanced incident summarization"""
        
        if self.api_key:
            try:
                prompt = f"""لخص هذا الوصف للحادثة في جملة واحدة واضحة ومباشرة:

"{description}"

التلخيص يجب أن يكون عملي ومفيد للإدارة."""

                response = self._call_llm(prompt, max_tokens=100, temperature=0.3)
                return response.strip()
                
            except Exception as e:
                self.logger.warning(f"LLM summarization failed: {e}")
        
        # Fallback summarization
        if len(description) <= 100:
            return description
        
        # Simple truncation with ellipsis
        return description[:97] + "..."

# Alias for backward compatibility
LLMService = EnhancedLLMServiceWithConfidence