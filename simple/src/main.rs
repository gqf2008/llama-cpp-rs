//! This is a translation of simple.cpp in llama.cpp using llama-cpp-2.
#![allow(
    clippy::cast_possible_wrap,
    clippy::cast_possible_truncation,
    clippy::cast_precision_loss,
    clippy::cast_sign_loss
)]

use anyhow::{anyhow, bail, Context, Result};
use clap::Parser;
use hf_hub::api::sync::ApiBuilder;
use llama_cpp_2::context::params::LlamaContextParams;

use llama_cpp_2::model::params::kv_overrides::ParamOverrideValue;
use llama_cpp_2::model::params::LlamaModelParams;
use llama_cpp_2::model::LlamaModel;
use std::ffi::CString;
use std::num::NonZeroU32;
use std::path::PathBuf;
use std::pin::pin;
use std::str::FromStr;

#[derive(clap::Parser, Debug, Clone)]
struct Args {
    /// The path to the model
    #[command(subcommand)]
    model: Model,
    /// The prompt
    #[clap(short = 'p', long)]
    prompt: Option<String>,
    /// Read the prompt from a file
    #[clap(short = 'f', long, help = "prompt file to start generation")]
    file: Option<String>,
    /// set the length of the prompt + output in tokens
    #[arg(long, default_value_t = 32)]
    n_len: i32,
    /// override some parameters of the model
    #[arg(short = 'o', value_parser = parse_key_val)]
    key_value_overrides: Vec<(String, ParamOverrideValue)>,
    /// Disable offloading layers to the gpu
    #[cfg(any(feature = "cuda", feature = "vulkan"))]
    #[clap(long)]
    disable_gpu: bool,
    #[arg(short = 's', long, help = "RNG seed (default: 1234)")]
    seed: Option<u32>,
    #[arg(
        short = 't',
        long,
        help = "number of threads to use during generation (default: use all available threads)"
    )]
    threads: Option<u32>,
    #[arg(
        long,
        help = "number of threads to use during batch and prompt processing (default: use all available threads)"
    )]
    threads_batch: Option<u32>,
    #[arg(
        short = 'c',
        long,
        help = "size of the prompt context (default: loaded from themodel)"
    )]
    ctx_size: Option<NonZeroU32>,
}

/// Parse a single key-value pair
fn parse_key_val(s: &str) -> Result<(String, ParamOverrideValue)> {
    let pos = s
        .find('=')
        .ok_or_else(|| anyhow!("invalid KEY=value: no `=` found in `{}`", s))?;
    let key = s[..pos].parse()?;
    let value: String = s[pos + 1..].parse()?;
    let value = i64::from_str(&value)
        .map(ParamOverrideValue::Int)
        .or_else(|_| f64::from_str(&value).map(ParamOverrideValue::Float))
        .or_else(|_| bool::from_str(&value).map(ParamOverrideValue::Bool))
        .map_err(|_| anyhow!("must be one of i64, f64, or bool"))?;

    Ok((key, value))
}

#[derive(clap::Subcommand, Debug, Clone)]
enum Model {
    /// Use an already downloaded model
    Local {
        /// The path to the model. e.g. `/home/marcus/.cache/huggingface/hub/models--TheBloke--Llama-2-7B-Chat-GGUF/blobs/08a5566d61d7cb6b420c3e4387a39e0078e1f2fe5f055f3a03887385304d4bfa`
        path: PathBuf,
    },
    /// Download a model from huggingface (or use a cached version)
    #[clap(name = "hf-model")]
    HuggingFace {
        /// the repo containing the model. e.g. `TheBloke/Llama-2-7B-Chat-GGUF`
        repo: String,
        /// the model name. e.g. `llama-2-7b-chat.Q4_K_M.gguf`
        model: String,
    },
}

impl Model {
    /// Convert the model to a path - may download from huggingface
    fn get_or_load(self) -> Result<PathBuf> {
        match self {
            Model::Local { path } => Ok(path),
            Model::HuggingFace { model, repo } => ApiBuilder::new()
                .with_progress(true)
                .build()
                .with_context(|| "unable to create huggingface api")?
                .model(repo)
                .get(&model)
                .with_context(|| "unable to download model"),
        }
    }
}

#[allow(clippy::too_many_lines)]
fn main() -> Result<()> {
    let Args {
        n_len,
        model,
        prompt,
        file,
        #[cfg(any(feature = "cuda", feature = "vulkan"))]
        disable_gpu,
        key_value_overrides,
        seed,
        threads,
        threads_batch,
        ctx_size,
    } = Args::parse();

    // offload all layers to the gpu
    let model_params = {
        #[cfg(any(feature = "cuda", feature = "vulkan"))]
        if !disable_gpu {
            LlamaModelParams::default().with_n_gpu_layers(1000)
        } else {
            LlamaModelParams::default()
        }
        #[cfg(not(any(feature = "cuda", feature = "vulkan")))]
        LlamaModelParams::default()
    };

    let prompt = if let Some(str) = prompt {
        if file.is_some() {
            bail!("either prompt or file must be specified, but not both")
        }
        str
    } else if let Some(file) = file {
        std::fs::read_to_string(&file).with_context(|| format!("unable to read {file}"))?
    } else {
        "Hello my name is".to_string()
    };

    let mut model_params = pin!(model_params);

    for (k, v) in &key_value_overrides {
        let k = CString::new(k.as_bytes()).with_context(|| format!("invalid key: {k}"))?;
        model_params.as_mut().append_kv_override(k.as_c_str(), *v);
    }

    let model_path = model
        .get_or_load()
        .with_context(|| "failed to get model from args")?;

    let model = LlamaModel::load_from_file(model_path, &model_params)
        .with_context(|| "unable to load model")?;

    let mut handles = vec![];
    // initialize the context
    let mut ctx_params = LlamaContextParams::default()
        .with_n_ctx(ctx_size.or(Some(NonZeroU32::new(4096).unwrap())))
        .with_seed(seed.unwrap_or(1234));
    if let Some(threads) = threads {
        ctx_params = ctx_params.with_n_threads(threads);
    }
    if let Some(threads_batch) = threads_batch.or(threads) {
        ctx_params = ctx_params.with_n_threads_batch(threads_batch);
    }

    for _ in 0..10 {
        let model = model.clone();
        let prompt = prompt.clone();
        let handle = std::thread::spawn(move || {
            for _ in 0..10 {
                let mut ctx = model
                    .clone()
                    .new_context(ctx_params.clone())
                    .with_context(|| "unable to create the llama_context")
                    .unwrap();
                let out = ctx.forward(prompt.clone(), n_len).unwrap();

                println!("{out}");
            }
        });
        handles.push(handle);
    }
    for h in handles {
        h.join().unwrap();
    }

    Ok(())
}
