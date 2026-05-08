from ollama import AsyncClient
import textwrap


class LegalGenerator:
    def __init__(self, host: str, model: str):
        self.client = AsyncClient(host=host)
        self.model = model

    def _build_answer_prompt(self, query: str, context: str, lang: str) -> str:
        """
        Constructs the RAG prompt in the requested language.
        """
        if lang == "en":
            return textwrap.dedent(f"""You are a strictly factual Bilingual Legal Assistant.
                Use ONLY the provided legal context to answer the user's question.
                Avoid repeating information in the final answer; if a point is covered in a list, do not repeat it in a summary note.
                If the context does not contain the answer, say exactly: "I cannot answer this based on the provided legal text."

                LEGAL CONTEXT:
                {context}

                USER QUESTION: 
                {query}

                ANSWER:""").strip()

        elif lang == "ar":
            return textwrap.dedent(f"""أنت مساعد قانوني دقيق.
                مهمتك هي استخراج الإجابة من النص القانوني التالي فقط.
                لا تضف أي معلومات من خارج النص. أجب باختصار شديد.

                النص القانوني:
                {context}

                السؤال:
                {query}

                الإجابة المستخرجة من النص:""").strip()
        else:
            raise ValueError(f"Unsupported language: {lang}")
        
    
    def _build_search_prompt(self, query: str, lang: str) -> str:
        if lang == "en":
            return textwrap.dedent(f"""You are an expert legal researcher. 
                Rewrite the user's input into a clear, standalone legal question optimized for a vector database.
                Remove conversational filler (like "Hi, can you tell me..."), but KEEP it as a natural language question.
                Output ONLY the revised question and nothing else.

                USER INPUT: {query}
                REVISED QUESTION:""").strip()

        elif lang == "ar":
            return textwrap.dedent(f"""أنت باحث قانوني خبير.
                أعد صياغة إدخال المستخدم إلى سؤال قانوني واضح ومستقل مُحسّن لقاعدة بيانات متجهة.
                قم بإزالة الحشو الحواري (مثل "مرحبًا، هل يمكنك إخباري...")، ولكن احتفظ به كسؤال باللغة الطبيعية.
                أخرج السؤال المُراجع فقط ولا شيء غيره.

                إدخال المستخدم: {query}
                السؤال المُراجع:""").strip()
        else:
            raise ValueError(f"Unsupported language: {lang}")
        

    async def generate_search_query(self, query: str, lang: str) -> str:
        prompt = self._build_search_prompt(query=query, lang=lang)

        try:
            res = await self.client.generate(
                model=self.model,
                prompt=prompt,
                options={"temperature": 0.0, "top_p": 0.1},
                think=False
            )

            return res["response"].strip().strip('"').strip("'")
        
        except Exception as e:
            print(f"Query generation failed: {e}")
            return query


    async def generate_answer(self, query: str, context: str, lang: str) -> str:
        prompt = self._build_answer_prompt(query=query, context=context, lang=lang)
        
        try:
            if lang == "en":
                response = await self.client.generate(
                    model=self.model,
                    prompt=prompt,
                    options={"temperature": 0.1, "top_p": 0.9},
                    think=False
                )
            else:
                response = await self.client.generate(
                    model=self.model,
                    prompt=prompt,
                    options={
                        "temperature": 0.2, 
                        "top_p": 0.9,
                        "num_predict": 1024,      
                        "repeat_penalty": 1.15  
                    },
                    think=False
                )
            return response['response']
        except Exception as e:
            return f"Error communicating with the generation model: {str(e)}"

