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

Example code could be found:

```python
import ollama
import chromadb

documents = [
    r"""copyrust document of each functions here""",
]

MODEL="mxbai-embed-large"

db_client = chromadb.Client()
ollama_client = ollama.Client(host='http://172.17.2.6:11434')
db_collection = db_client.create_collection(name="rust_docs")

for i, d in enumerate(documents):
    response = ollama_client.embed(model=MODEL, input=d)
    embeddings = response["embeddings"]
    db_collection.add(ids=[str(i)], embeddings=embeddings, documents=[d])



def get_reply(question):
    response = ollama_client.embed(
      model=MODEL,
      input=question
    )

    results = db_collection.query(
      query_embeddings=response["embeddings"],
      n_results=1
    )
    return results['documents'][0][0]

print(get_reply("how to sort a Vec"))
```

TODO: passing RAG results to LLM

Further reading: https://techcommunity.microsoft.com/blog/azuredevcommunityblog/doing-rag-vector-search-is-not-enough/4161073

## Conclusion

For Rust coding, Since I already have [great rust document searcher][1], AI is
only slowing me down. Spending time to generate data for RAG will not improve
this.

For non-Rust coding, I will use AI as quicker search engine than google.

Enterprise company could use RAG(with or without LLM) to provide 100% accurate
suggestions without any AI delusion.

[1]: $HOME/.rustup/toolchains/nightly-x86_64-unknown-linux-gnu/share/doc/rust/html/std/index.html
    after `rustup component add rust-docs-x86_64-unknown-linux-gnu`
    and `cargo docs` for non-std crates

[2]: https://github.com/ollama/ollama/blob/main/docs/api.md#generate-embeddings
[rag_screencast]: internal_only
