"""Dense embedding service using multilingual E5.

The Sentence Transformers dependency is loaded lazily so importing this module
does not immediately require PyTorch to initialize. This is especially useful
for development environments where the embedding runtime is unavailable but
the rest of the retrieval pipeline can still be developed and tested.
"""

from collections.abc import Sequence
from typing import Any

from src.embeddings.base import EmbeddingService, EmbeddingVector


MODEL_NAME = "intfloat/multilingual-e5-large-instruct"

# The selected E5 model produces 1024-dimensional embeddings. The runtime
# dimension is checked when the model is eventually loaded so configuration
# errors fail early rather than reaching Qdrant.
EXPECTED_DIMENSION = 1024

# E5-instruct models distinguish retrieval queries from document passages.
# Queries receive an explicit retrieval instruction while passages remain
# unmodified before encoding.
QUERY_INSTRUCTION = (
    "Retrieve relevant official Nepal government passages "
    "that answer the user's question."
)


class E5EmbeddingService(EmbeddingService):
    """Embedding provider backed by multilingual-e5-large-instruct."""

    def __init__(
        self,
        model_name: str = MODEL_NAME,
        device: str | None = None,
        batch_size: int = 16,
    ) -> None:
        """Configure the service without loading the model immediately.

        Lazy loading is deliberate: application modules can construct and
        inspect the service without initializing PyTorch until embeddings are
        actually requested.
        """

        if batch_size <= 0:
            raise ValueError(
                "batch_size must be greater than zero."
            )

        self.model_name = model_name
        self.device = device
        self.batch_size = batch_size

        # Avoid importing or initializing Sentence Transformers until an
        # embedding operation actually needs the model.
        self._model: Any | None = None

    @property
    def dimension(self) -> int:
        """Return the expected vector dimension used by the Qdrant schema."""

        return EXPECTED_DIMENSION

    @staticmethod
    def format_query(query: str) -> str:
        """Format a user query according to the E5-instruct convention."""

        clean_query = query.strip()

        if not clean_query:
            raise ValueError(
                "Query must contain non-whitespace text."
            )

        # Keep the retrieval task instruction separate from the user's actual
        # query. This follows the intended instruction/query structure for the
        # selected E5-instruct model.
        return (
            f"Instruct: {QUERY_INSTRUCTION}\n"
            f"Query: {clean_query}"
        )

    @staticmethod
    def _validate_passages(
        texts: Sequence[str],
    ) -> list[str]:
        """Validate and normalize passage inputs before model execution."""

        if not texts:
            raise ValueError(
                "At least one passage is required."
            )

        cleaned_texts: list[str] = []

        for index, text in enumerate(texts):
            clean_text = text.strip()

            if not clean_text:
                raise ValueError(
                    f"Passage at index {index} is empty."
                )

            cleaned_texts.append(clean_text)

        return cleaned_texts

    def _load_model(self) -> Any:
        """Load Sentence Transformers only when embeddings are requested."""

        if self._model is not None:
            return self._model

        try:
            # Import locally instead of at module level. On the current Windows
            # machine, Smart App Control blocks native PyTorch DLLs, but that
            # should not prevent the rest of NepalGov AI from importing this
            # embedding service.
            from sentence_transformers import SentenceTransformer

            model_kwargs: dict[str, Any] = {}

            # Let Sentence Transformers select the device automatically unless
            # a runtime such as "cpu" or "cuda" is explicitly requested.
            if self.device is not None:
                model_kwargs["device"] = self.device

            self._model = SentenceTransformer(
                self.model_name,
                **model_kwargs,
            )

        except (ImportError, OSError) as exc:
            raise RuntimeError(
                "Unable to initialize the E5 embedding runtime. "
                "The current environment must support Sentence Transformers "
                "and PyTorch before embeddings can be generated."
            ) from exc

        runtime_dimension = (
            self._model.get_sentence_embedding_dimension()
        )

        if runtime_dimension != EXPECTED_DIMENSION:
            raise RuntimeError(
                "Unexpected embedding dimension: "
                f"expected {EXPECTED_DIMENSION}, "
                f"received {runtime_dimension}."
            )

        return self._model

    def embed_passages(
        self,
        texts: Sequence[str],
    ) -> list[EmbeddingVector]:
        """Create normalized embeddings for document passages."""

        passages = self._validate_passages(
            texts
        )

        model = self._load_model()

        # Cosine search works cleanly with normalized embeddings and keeps the
        # embedding convention consistent for passages and queries.
        embeddings = model.encode(
            passages,
            batch_size=self.batch_size,
            normalize_embeddings=True,
            convert_to_numpy=True,
            show_progress_bar=False,
        )

        return embeddings.tolist()

    def embed_query(
        self,
        query: str,
    ) -> EmbeddingVector:
        """Create one normalized embedding for a retrieval query."""

        formatted_query = self.format_query(
            query
        )

        model = self._load_model()

        # Encode as a one-item batch so the implementation uses the same model
        # pathway as passage encoding and returns a predictable vector shape.
        embedding = model.encode(
            [formatted_query],
            batch_size=1,
            normalize_embeddings=True,
            convert_to_numpy=True,
            show_progress_bar=False,
        )

        return embedding[0].tolist()