from transformers import pipeline


pipe = pipeline(
    "text-generation",
    model="microsoft/phi-2"
)


class LocalLLM:

    def invoke(self, prompt):

        result = pipe(
            prompt,
            max_new_tokens=200,
            truncation=True
        )

        text = result[0]["generated_text"]

        # Remove original prompt from output
        return text.replace(prompt, "").strip()


llm = LocalLLM()