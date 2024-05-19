from transformers import BertConfig, BertModel

# Download model and configuration from huggingface.co and cache.
model = BertModel.from_pretrained("bert-base-uncased")
# Model was saved using *save_pretrained('./test/saved_model/')* (for example purposes, not runnable).
model = BertModel.from_pretrained("./test/saved_model/")
# Update configuration during loading.
model = BertModel.from_pretrained("bert-base-uncased", output_attentions=True)
assert model.config.output_attentions == True
# Loading from a TF checkpoint file instead of a PyTorch model (slower, for example purposes, not runnable).
config = BertConfig.from_json_file("./tf_model/my_tf_model_config.json")
model = BertModel.from_pretrained("./tf_model/my_tf_checkpoint.ckpt.index", from_tf=True, config=config)
# Loading from a Flax checkpoint file instead of a PyTorch model (slower)
model = BertModel.from_pretrained("bert-base-uncased", from_flax=True)