# SPDX-License-Identifier: Apache-2.0

import ollama
import chromadb

documents = [
    r"""
    Rust document for Vec::sort():

    Sorts the slice with a comparison function, preserving initial order of equal elements.

This sort is stable (i.e., does not reorder equal elements) and *O*(*n* \* log(*n*))
worst-case.

If the comparison function `compare` does not implement a [total order], the function may
panic; even if the function exits normally, the resulting order of elements in the slice is
unspecified. See also the note on panicking below.

For example `|a, b| (a - b).cmp(a)` is a comparison function that is neither transitive nor
reflexive nor total, `a < b < c < a` with `a = 1, b = 2, c = 3`. For more information and
examples see the [`Ord`] documentation.

# Current implementation

The current implementation is based on [driftsort] by Orson Peters and Lukas Bergdoll, which
combines the fast average case of quicksort with the fast worst case and partial run
detection of mergesort, achieving linear time on fully sorted and reversed inputs. On inputs
with k distinct elements, the expected time to sort the data is *O*(*n* \* log(*k*)).

The auxiliary memory allocation behavior depends on the input length. Short slices are
handled without allocation, medium sized slices allocate `self.len()` and beyond that it
clamps at `self.len() / 2`.

# Panics

May panic if `compare` does not implement a [total order], or if `compare` itself panics.

All safe functions on slices preserve the invariant that even if the function panics, all
original elements will remain in the slice and any possible modifications via interior
mutability are observed in the input. This ensures that recovery code (for instance inside
of a `Drop` or following a `catch_unwind`) will still have access to all the original
elements. For instance, if the slice belongs to a `Vec`, the `Vec::drop` method will be able
to dispose of all contained elements.

# Examples

```
let mut v = [4, -5, 1, -3, 2];
v.sort_by(|a, b| a.cmp(b));
assert_eq!(v, [-5, -3, 1, 2, 4]);

// reverse sorting
v.sort_by(|a, b| b.cmp(a));
assert_eq!(v, [4, 2, 1, -3, -5]);
```

[driftsort]: https://github.com/Voultapher/driftsort
[total order]: https://en.wikipedia.org/wiki/Total_order
""",
    r"""
The rust document for Vec::sort_unstable():

Sorts the slice with a comparison function, **without** preserving the initial order of
equal elements.
This sort is unstable (i.e., may reorder equal elements), in-place (i.e., does not
allocate), and *O*(*n* \* log(*n*)) worst-case.
If the comparison function `compare` does not implement a [total order], the function
may panic; even if the function exits normally, the resulting order of elements in the slice
is unspecified. See also the note on panicking below.
For example `|a, b| (a - b).cmp(a)` is a comparison function that is neither transitive nor
reflexive nor total, `a < b < c < a` with `a = 1, b = 2, c = 3`. For more information and
examples see the [`Ord`] documentation.
All original elements will remain in the slice and any possible modifications via interior
mutability are observed in the input. Same is true if `compare` panics.
# Current implementation
The current implementation is based on [ipnsort] by Lukas Bergdoll and Orson Peters, which
combines the fast average case of quicksort with the fast worst case of heapsort, achieving
linear time on fully sorted and reversed inputs. On inputs with k distinct elements, the
expected time to sort the data is *O*(*n* \* log(*k*)).
It is typically faster than stable sorting, except in a few special cases, e.g., when the
slice is partially sorted.
# Panics
May panic if the `compare` does not implement a [total order], or if
the `compare` itself panics.
# Examples
```
let mut v = [4, -5, 1, -3, 2];
v.sort_unstable_by(|a, b| a.cmp(b));
assert_eq!(v, [-5, -3, 1, 2, 4]);
// reverse sorting
v.sort_unstable_by(|a, b| b.cmp(a));
assert_eq!(v, [4, 2, 1, -3, -5]);
```
[ipnsort]: https://github.com/Voultapher/sort-research-rs/tree/main/ipnsort
[total order]: https://en.wikipedia.org/wiki/Total_order
            """,
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
