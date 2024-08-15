//! Safe wrapper around `llama_context`.

use std::fmt::{Debug, Formatter};
use std::io::Write;
use std::num::NonZeroI32;
use std::ptr::NonNull;
use std::slice;
use std::time::Duration;

use crate::llama_batch::LlamaBatch;
use crate::model::{AddBos, LlamaLoraAdapter, LlamaModel, Special};
use crate::timing::LlamaTimings;
use crate::token::data::LlamaTokenData;
use crate::token::data_array::LlamaTokenDataArray;
use crate::token::LlamaToken;
use crate::{
    DecodeError, EmbeddingsError, EncodeError, LlamaLoraAdapterRemoveError,
    LlamaLoraAdapterSetError,
};

pub mod kv_cache;
pub mod params;
pub mod sample;
pub mod session;

/// Safe wrapper around `llama_context`.
#[allow(clippy::module_name_repetitions)]
pub struct LlamaContext {
    pub(crate) context: NonNull<llama_cpp_sys_2::llama_context>,
    initialized_logits: Vec<i32>,
    embeddings_enabled: bool,
    /// model
    pub model: LlamaModel,
}

impl Debug for LlamaContext {
    fn fmt(&self, f: &mut Formatter) -> std::fmt::Result {
        f.debug_struct("LlamaContext")
            .field("context", &self.context)
            .finish()
    }
}

impl LlamaContext {
    pub(crate) fn new(
        model: LlamaModel,
        llama_context: NonNull<llama_cpp_sys_2::llama_context>,
        embeddings_enabled: bool,
    ) -> Self {
        Self {
            model,
            context: llama_context,
            initialized_logits: Vec::new(),
            embeddings_enabled,
        }
    }

    /// Gets the max number of tokens in a batch.
    #[must_use]
    pub fn n_batch(&self) -> u32 {
        unsafe { llama_cpp_sys_2::llama_n_batch(self.context.as_ptr()) }
    }

    /// Gets the size of the context.
    #[must_use]
    pub fn n_ctx(&self) -> u32 {
        unsafe { llama_cpp_sys_2::llama_n_ctx(self.context.as_ptr()) }
    }

    /// Decodes the batch.
    ///
    /// # Errors
    ///
    /// - `DecodeError` if the decoding failed.
    ///
    /// # Panics
    ///
    /// - the returned [`std::ffi::c_int`] from llama-cpp does not fit into a i32 (this should never happen on most systems)
    pub fn decode(&mut self, batch: &mut LlamaBatch) -> Result<(), DecodeError> {
        let result =
            unsafe { llama_cpp_sys_2::llama_decode(self.context.as_ptr(), batch.llama_batch) };

        match NonZeroI32::new(result) {
            None => {
                self.initialized_logits
                    .clone_from(&batch.initialized_logits);
                Ok(())
            }
            Some(error) => Err(DecodeError::from(error)),
        }
    }

    /// Encodes the batch.
    ///
    /// # Errors
    ///
    /// - `EncodeError` if the decoding failed.
    ///
    /// # Panics
    ///
    /// - the returned [`std::ffi::c_int`] from llama-cpp does not fit into a i32 (this should never happen on most systems)
    pub fn encode(&mut self, batch: &mut LlamaBatch) -> Result<(), EncodeError> {
        let result =
            unsafe { llama_cpp_sys_2::llama_encode(self.context.as_ptr(), batch.llama_batch) };

        match NonZeroI32::new(result) {
            None => {
                self.initialized_logits
                    .clone_from(&batch.initialized_logits);
                Ok(())
            }
            Some(error) => Err(EncodeError::from(error)),
        }
    }

    /// Get the embeddings for the `i`th sequence in the current context.
    ///
    /// # Returns
    ///
    /// A slice containing the embeddings for the last decoded batch.
    /// The size corresponds to the `n_embd` parameter of the context's model.
    ///
    /// # Errors
    ///
    /// - When the current context was constructed without enabling embeddings.
    /// - If the current model had a pooling type of [`llama_cpp_sys_2::LLAMA_POOLING_TYPE_NONE`]
    /// - If the given sequence index exceeds the max sequence id.
    ///
    /// # Panics
    ///
    /// * `n_embd` does not fit into a usize
    pub fn embeddings_seq_ith(&self, i: i32) -> Result<&[f32], EmbeddingsError> {
        if !self.embeddings_enabled {
            return Err(EmbeddingsError::NotEnabled);
        }

        let n_embd =
            usize::try_from(self.model.n_embd()).expect("n_embd does not fit into a usize");

        unsafe {
            let embedding = llama_cpp_sys_2::llama_get_embeddings_seq(self.context.as_ptr(), i);

            // Technically also possible whenever `i >= max(batch.n_seq)`, but can't check that here.
            if embedding.is_null() {
                Err(EmbeddingsError::NonePoolType)
            } else {
                Ok(slice::from_raw_parts(embedding, n_embd))
            }
        }
    }

    /// Get the embeddings for the `i`th token in the current context.
    ///
    /// # Returns
    ///
    /// A slice containing the embeddings for the last decoded batch of the given token.
    /// The size corresponds to the `n_embd` parameter of the context's model.
    ///
    /// # Errors
    ///
    /// - When the current context was constructed without enabling embeddings.
    /// - When the given token didn't have logits enabled when it was passed.
    /// - If the given token index exceeds the max token id.
    ///
    /// # Panics
    ///
    /// * `n_embd` does not fit into a usize
    pub fn embeddings_ith(&self, i: i32) -> Result<&[f32], EmbeddingsError> {
        if !self.embeddings_enabled {
            return Err(EmbeddingsError::NotEnabled);
        }

        let n_embd =
            usize::try_from(self.model.n_embd()).expect("n_embd does not fit into a usize");

        unsafe {
            let embedding = llama_cpp_sys_2::llama_get_embeddings_ith(self.context.as_ptr(), i);
            // Technically also possible whenever `i >= batch.n_tokens`, but no good way of checking `n_tokens` here.
            if embedding.is_null() {
                Err(EmbeddingsError::LogitsNotEnabled)
            } else {
                Ok(slice::from_raw_parts(embedding, n_embd))
            }
        }
    }

    /// Get the logits for the ith token in the context.
    ///
    /// # Panics
    ///
    /// - logit `i` is not initialized.
    pub fn candidates_ith(&self, i: i32) -> impl Iterator<Item = LlamaTokenData> + '_ {
        (0_i32..).zip(self.get_logits_ith(i)).map(|(i, logit)| {
            let token = LlamaToken::new(i);
            LlamaTokenData::new(token, *logit, 0_f32)
        })
    }

    /// Get the logits for the ith token in the context.
    ///
    /// # Panics
    ///
    /// - `i` is greater than `n_ctx`
    /// - `n_vocab` does not fit into a usize
    /// - logit `i` is not initialized.
    #[must_use]
    pub fn get_logits_ith(&self, i: i32) -> &[f32] {
        assert!(
            self.initialized_logits.contains(&i),
            "logit {i} is not initialized. only {:?} is",
            self.initialized_logits
        );
        assert!(
            self.n_ctx() > u32::try_from(i).expect("i does not fit into a u32"),
            "n_ctx ({}) must be greater than i ({})",
            self.n_ctx(),
            i
        );

        let data = unsafe { llama_cpp_sys_2::llama_get_logits_ith(self.context.as_ptr(), i) };
        let len = usize::try_from(self.model.n_vocab()).expect("n_vocab does not fit into a usize");

        unsafe { slice::from_raw_parts(data, len) }
    }

    /// Reset the timings for the context.
    pub fn reset_timings(&mut self) {
        unsafe { llama_cpp_sys_2::llama_reset_timings(self.context.as_ptr()) }
    }

    /// Returns the timings for the context.
    pub fn timings(&mut self) -> LlamaTimings {
        let timings = unsafe { llama_cpp_sys_2::llama_get_timings(self.context.as_ptr()) };
        LlamaTimings { timings }
    }

    /// Sets a lora adapter.
    ///
    /// # Errors
    ///
    /// See [`LlamaLoraAdapterSetError`] for more information.
    pub fn lora_adapter_set(
        &self,
        adapter: &mut LlamaLoraAdapter,
        scale: f32,
    ) -> Result<(), LlamaLoraAdapterSetError> {
        let err_code = unsafe {
            llama_cpp_sys_2::llama_lora_adapter_set(
                self.context.as_ptr(),
                adapter.lora_adapter.as_ptr(),
                scale,
            )
        };
        if err_code != 0 {
            return Err(LlamaLoraAdapterSetError::ErrorResult(err_code));
        }

        tracing::debug!("Set lora adapter");
        Ok(())
    }

    /// Remove a lora adapter.
    ///
    /// # Errors
    ///
    /// See [`LlamaLoraAdapterRemoveError`] for more information.
    pub fn lora_adapter_remove(
        &self,
        adapter: &mut LlamaLoraAdapter,
    ) -> Result<(), LlamaLoraAdapterRemoveError> {
        let err_code = unsafe {
            llama_cpp_sys_2::llama_lora_adapter_remove(
                self.context.as_ptr(),
                adapter.lora_adapter.as_ptr(),
            )
        };
        if err_code != 0 {
            return Err(LlamaLoraAdapterRemoveError::ErrorResult(err_code));
        }

        tracing::debug!("Remove lora adapter");
        Ok(())
    }
}

impl LlamaContext {
    /// forward
    pub fn forward<S: AsRef<str>>(&mut self, prompt: S, max_length: i32) -> anyhow::Result<String> {
        let tokens_list = self.model.str_to_token(prompt.as_ref(), AddBos::Always)?;
        let n_cxt = self.n_ctx() as i32;
        let n_kv_req = tokens_list.len() as i32 + (max_length - tokens_list.len() as i32);
        log::debug!("max_length = {max_length}, n_ctx = {n_cxt}, k_kv_req = {n_kv_req}");
        if n_kv_req > n_cxt {
            anyhow::bail!(
                "n_kv_req > n_ctx, the required kv cache size is not big enough
    either reduce n_len or increase n_ctx"
            )
        }

        if tokens_list.len() >= usize::try_from(max_length)? {
            anyhow::bail!("the prompt is too long, it has more tokens than max_length")
        }
        self.clear_kv_cache();
        let mut batch = LlamaBatch::new(tokens_list.len(), 1);

        let last_index: i32 = (tokens_list.len() - 1) as i32;
        for (i, token) in (0_i32..).zip(tokens_list.into_iter()) {
            let is_last = i == last_index;
            batch.add(token, i, &[0], is_last)?;
        }
        self.decode(&mut batch)?;
        let mut n_cur = batch.n_tokens();
        let mut n_decode = 0;

        let t_main_start = crate::ggml_time_us();
        let mut output = String::new();
        let mut decoder = encoding_rs::UTF_8.new_decoder();

        while n_cur <= max_length {
            {
                let candidates = self.candidates_ith(batch.n_tokens() - 1);

                let candidates_p = LlamaTokenDataArray::from_iter(candidates, false);

                // sample the most likely token
                let new_token_id = self.sample_token_greedy(candidates_p);

                // is it an end of stream?
                if new_token_id == self.model.token_eos() || new_token_id == self.model.token_eot()
                {
                    // eprintln!();
                    break;
                }

                let output_bytes = self.model.token_to_bytes(new_token_id, Special::Tokenize)?;
                // use `Decoder.decode_to_string()` to avoid the intermediate buffer
                let mut output_string = String::with_capacity(32);
                let _decode_result =
                    decoder.decode_to_string(&output_bytes, &mut output_string, false);
                output.push_str(output_string.as_str());
                batch.clear();
                batch.add(new_token_id, n_cur, &[0], true)?;
            }

            n_cur += 1;

            self.decode(&mut batch)?;

            n_decode += 1;
        }
        log::debug!("{output}");

        let t_main_end = crate::ggml_time_us();
        let duration = Duration::from_micros((t_main_end - t_main_start) as u64);
        log::debug!(
            "decoded {} tokens in {:.2} s, speed {:.2} t/s timings {}\n",
            n_decode,
            duration.as_secs_f32(),
            n_decode as f32 / duration.as_secs_f32(),
            self.timings()
        );
        Ok(output)
    }
}

impl Drop for LlamaContext {
    fn drop(&mut self) {
        unsafe { llama_cpp_sys_2::llama_free(self.context.as_ptr()) }
    }
}
