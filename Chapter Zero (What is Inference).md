The lifecycle of a generative AI model can be split into two parts:
1. Training
2. Inference.

Inference, here, is the art of serving the model in production. It is a very complex process. Three layers are required to do inference:

1. **Runtime** (which is optimizing the performance of a single model on a single GPU-backed instance)
2. **Infrastructure:** Scaling that model across clusters, regions and clouds while maintaining excellent uptime. 
3. **Tooling** Giving the engineers working on inference the right level of abstraction to work on these tasks.

Some choice terms:

### Batching
Running incoming requests in parallel.
### Caching
Reusing the KV cache to reduce wasteful computation
### Quantization
Lowering the precision of parts of the model to reduce memory burden=
### Speculation
Generating draft tokens to produce more than one token per forward pass during the decoding phase of the model.
### Parallelism
Using more GPUs to accelerate the model without causing bottlenecks.
### Disaggregation
Separating the two phases of LLM inference (prefill and decode) to workers that can be scaled differently.