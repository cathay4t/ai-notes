## AI helping coding

Task:
 * Given 2 sound sinks(HDMI screen audio, USB headset),
 * When I run a scirpt
 * Then the default sound sink switch to another one

When AI is actual helping:
 * Provide suggestion on which command or argument for certain task.
 * Example code on frequently asked questions when this is not your daily
   coding language, for examples:
    * How to sort in python by object key?
    * How to exit in python with non-zero code?

When AI not preforming well:
 * Since most developer on this topic is using bash with regex parsing, AI
   has no pre-trained data on JSON handling of `pw-dump`.
 * Logic on filter information of `pw-dump`.

## Limit AI reply by Retrieval-Augmented Generation

Please check this [screencast][rag_screencast] to understand how RAG works by
using `chromadb` and [ollama embedding][2].

The code is based on the work https://ollama.com/blog/embedding-models with
small fixes on typo and API changes.

Example code could be found at https://github.com/cathay4t/ai-notes/blob/main/rag.py

RAG will search out the most related pre-defined documents based on your
questions, then you can redirect the these documents to LLM seeking for better
replies.

Further reading: https://techcommunity.microsoft.com/blog/azuredevcommunityblog/doing-rag-vector-search-is-not-enough/4161073

## Conclusion

For Rust coding, Since I already have [great rust document searcher][1], AI is
only slowing me down. Spending time to generate data for RAG will not improve
this.

For non-Rust coding, I will use AI as quicker search engine than google.

Enterprise company could use RAG(with or without LLM) to provide 100% accurate
suggestions without any AI delusion.

[1]: After `rustup component add rust-docs-x86_64-unknown-linux-gnu`
and `cargo docs` for non-std crates

[2]: https://github.com/ollama/ollama/blob/main/docs/api.md#generate-embeddings
[rag_screencast]: internal_only
