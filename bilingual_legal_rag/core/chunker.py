class ChunkingManager:
    def __init__(self, model, context_len, overlap_tokens=50):
        self.model = model
        self.context_len = max(0, context_len - 50)
        self.overlap_tokens = overlap_tokens

    def _get_separators(self, doc_type: str):
        separator_dict = {
            "prose": ["\n\n", "\n", ". ", " "],
            "markdown": ["\n## ", "\n### ", "\n#### ", "\n\n", "\n", ". ", " "],
            "python": ["\nclass ", "\ndef ", "\n\n", "\n", " "],
            "html": ["</section>", "</div>", "</p>", "\n\n", "\n", ">", " "],
            "shell": ["\nfunction ", "\n\n# ", "\n\n", "\n", " "],
            "log": ["\n20", "\n19", "\n[", "\n\n", "\n"],
            "arabic-legal": None,
            "english-legal": ["\n\n", ";\n\n", "\n", ". ", " "],
        }
        
        return separator_dict.get(doc_type, separator_dict["prose"])

    def _count_tokens(self, text: str) -> int:
        if not text or not text.strip():
            return 0
        return len(self.model.tokenize(text.encode("utf-8")))

    def _get_overlap(self, text: str) -> str:
        if not text or self.overlap_tokens == 0:
            return ""
        tokens = self.model.tokenize(text.encode("utf-8"))
        overlap_tokens = tokens[-self.overlap_tokens:]
        return self.model.detokenize(overlap_tokens).decode("utf-8", errors="replace")

    def _chunk_rec(self, text: str, separators: list, sep_idx: int) -> list[str]:
        if self._count_tokens(text) <= self.context_len:
            return [text]

        if sep_idx >= len(separators):
            return self._fallback_chunk(text)

        sep = separators[sep_idx]
        raw_pieces = text.split(sep)
        pieces = [raw_pieces[0]] + [sep + p for p in raw_pieces[1:]]

        chunks = []
        current_chunk = ""
        token_cnt = 0

        for piece in pieces:
            piece_tokens = self._count_tokens(piece)

            if piece_tokens > self.context_len:
                if current_chunk:
                    chunks.append(current_chunk.strip())
                    current_chunk = ""
                    token_cnt = 0
                deep_chunks = self._chunk_rec(piece, separators, sep_idx + 1)
                chunks.extend(deep_chunks)
                continue

            if token_cnt + piece_tokens > self.context_len:
                chunks.append(current_chunk.strip())
                overlap = self._get_overlap(current_chunk)
                current_chunk = overlap + piece
                token_cnt = self._count_tokens(current_chunk)
            else:
                current_chunk += piece
                token_cnt += piece_tokens

        if current_chunk:
            chunks.append(current_chunk.strip())

        return [c for c in chunks if c]

    def _fallback_chunk(self, text: str) -> list[str]:
        tokens = self.model.tokenize(text.encode("utf-8"))
        chunks = []
        for i in range(0, len(tokens), self.context_len):
            chunk_tokens = tokens[i:i + self.context_len]
            chunk = self.model.detokenize(chunk_tokens).decode("utf-8", errors="replace")
            if chunk.strip():
                chunks.append(chunk)
        return chunks

    def chunk_doc(self, doc: str, doc_type="prose"):
        if not doc.strip():
            return []
        separators = self._get_separators(doc_type)
        return self._chunk_rec(doc, separators, 0)