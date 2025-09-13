from typing import Dict, Any

import torch
from transformers import pipeline, AutoTokenizer, AutoModelForCausalLM

from config import Config


class LocalLLM:
    def __init__(self):
        """Initialize LLM (OpenAI or local)"""
        self.use_openai = Config.USE_OPENAI

        if self.use_openai:
            import openai
            openai.api_key = Config.OPENAI_API_KEY
            self.client = openai.OpenAI()
            self.model_name = Config.OPENAI_LLM_MODEL
            self.pipeline = None
        else:
            self.model_name = Config.LOCAL_LLM_MODEL
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
            self._load_local_model()

    def _load_local_model(self):
        """Load local transformer model"""
        try:
            # Load tokenizer and model
            self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
            self.model = AutoModelForCausalLM.from_pretrained(
                self.model_name,
                dtype=torch.float16 if self.device == "cuda" else torch.float32,
                device_map="auto" if self.device == "cuda" else None,
                low_cpu_mem_usage=True
            )

            # Create pipeline
            self.pipeline = pipeline(
                "text-generation",
                model=self.model,
                tokenizer=self.tokenizer,
                device=0 if self.device == "cuda" else -1
            )

            # Set pad token if not exists
            if self.tokenizer.pad_token is None:
                self.tokenizer.pad_token = self.tokenizer.eos_token

        except Exception as e:
            print(f"Error loading model {self.model_name}: {e}")
            # Fallback to a smaller model
            self.model_name = "distilgpt2"
            self._load_fallback_model()

    def _load_fallback_model(self):
        """Load fallback model if primary model fails"""
        try:
            self.pipeline = pipeline(
                "text-generation",
                model=self.model_name,
                device=0 if self.device == "cuda" else -1
            )
        except Exception as e:
            print(f"Error loading fallback model: {e}")
            self.pipeline = None

    def generate_answer(self, query: str, context: str) -> Dict[str, Any]:
        """Generate answer using RAG prompt template"""
        if self.use_openai:
            return self._generate_openai_answer(query, context)
        else:
            return self._generate_local_answer(query, context)

    def _generate_openai_answer(self, query: str, context: str) -> Dict[str, Any]:
        """Generate answer using OpenAI API"""
        try:
            response = self.client.chat.completions.create(
                model=self.model_name,
                messages=[
                    {"role": "system",
                     "content": "You are an internal research assistant. Use ONLY the provided context to answer the query. If insufficient information, say 'Insufficient data.'"},
                    {"role": "user", "content": f"Query: {query}\n\nContext:\n{context}\n\nAnswer:"}
                ],
                max_tokens=300,
                temperature=0.7
            )

            answer = response.choices[0].message.content.strip()

            return {
                'answer': answer,
                'model_used': self.model_name,
                'error': False
            }

        except Exception as e:
            return {
                'answer': f"Error generating response: {str(e)}",
                'model_used': self.model_name,
                'error': True
            }

    def _generate_local_answer(self, query: str, context: str) -> Dict[str, Any]:
        """Generate answer using local model"""
        if self.pipeline is None:
            return {
                'answer': "LLM not available. Please check model installation.",
                'model_used': "none",
                'error': True
            }

        # RAG prompt template
        prompt = f"""You are an internal research assistant. Use ONLY the provided context to answer the query. If insufficient information, say "Insufficient data."

Query: {query}

Context:
{context}

Answer:"""

        try:
            # Generate response
            response = self.pipeline(
                prompt,
                max_length=len(prompt.split()) + 150,  # Limit response length
                num_return_sequences=1,
                temperature=0.7,
                do_sample=True,
                pad_token_id=self.pipeline.tokenizer.eos_token_id
            )

            # Extract generated text
            generated_text = response[0]['generated_text']

            # Extract only the answer part
            answer_start = generated_text.find("Answer:") + len("Answer:")
            answer = generated_text[answer_start:].strip()

            # Clean up the answer
            answer = self._clean_answer(answer)

            return {
                'answer': answer,
                'model_used': self.model_name,
                'error': False
            }

        except Exception as e:
            return {
                'answer': f"Error generating response: {str(e)}",
                'model_used': self.model_name,
                'error': True
            }

    def _clean_answer(self, answer: str) -> str:
        """Clean and format the generated answer"""
        # Remove any repeated prompt text
        lines = answer.split('\n')
        cleaned_lines = []

        for line in lines:
            line = line.strip()
            if line and not line.startswith(('Query:', 'Context:', 'Answer:')):
                cleaned_lines.append(line)

        cleaned_answer = '\n'.join(cleaned_lines)

        # Limit length
        if len(cleaned_answer) > 500:
            cleaned_answer = cleaned_answer[:500] + "..."

        return cleaned_answer if cleaned_answer else "No answer generated."

    def get_model_info(self) -> Dict[str, Any]:
        """Get information about the loaded model"""
        if self.use_openai:
            return {
                'model_name': self.model_name,
                'device': 'OpenAI API',
                'available': Config.OPENAI_API_KEY is not None
            }
        else:
            return {
                'model_name': self.model_name,
                'device': self.device,
                'available': self.pipeline is not None
            }
