import streamlit as st
import numpy as np
import glob
import os
import json
import torch
import matplotlib.pyplot as plt
import struct
import pickle
from PIL import Image

st.set_page_config(page_title="FluidVLA Explorer", layout="wide", initial_sidebar_state="expanded")

st.title("🌊 FluidVLA Explorer")
st.markdown("Explore datasets, model checkpoints, and training logs.")

# --- NAVIGATION ---
mode = st.sidebar.radio("Mode", ["Dataset Explorer (.npz, .npy, raw)", "Model & Logs Explorer (.pt, .json)"])

st.sidebar.markdown("---")

# ----------------------------------------------------
# Custom Loaders for MNIST and CIFAR-10
# ----------------------------------------------------
def read_idx(filename):
    """Read MNIST idx files (idx3-ubyte or idx1-ubyte)"""
    import gzip
    open_fn = gzip.open if filename.endswith('.gz') else open
    with open_fn(filename, 'rb') as f:
        zero, data_type, dims = struct.unpack('>HBB', f.read(4))
        shape = tuple(struct.unpack('>I', f.read(4))[0] for d in range(dims))
        return np.frombuffer(f.read(), dtype=np.uint8).reshape(shape)

def load_cifar_batch(file):
    """Read CIFAR-10 data_batch_* files"""
    with open(file, 'rb') as fo:
        dict_data = pickle.load(fo, encoding='bytes')
    # keys are usually b'batch_label', b'labels', b'data', b'filenames'
    data = dict_data[b'data']
    labels = dict_data[b'labels']
    data = data.reshape(-1, 3, 32, 32).transpose(0, 2, 3, 1) # to (N, H, W, 3)
    return data, labels

def load_cifar_meta(file):
    with open(file, 'rb') as fo:
        dict_data = pickle.load(fo, encoding='bytes')
    return [label.decode('utf-8') for label in dict_data[b'label_names']]

# ----------------------------------------------------
# Page Logic
# ----------------------------------------------------

if mode == "Dataset Explorer (.npz, .npy, raw)":
    st.header("📂 Dataset Explorer")
    
    # Path selection
    data_dir_options = ["./data/step2", "./data/step2_isaac"]
    
    # Auto-discover other subdirectories in ./data/
    if os.path.exists("./data"):
        for item in os.listdir("./data"):
            item_path = os.path.join(".", "data", item)
            if os.path.isdir(item_path) and item_path not in data_dir_options:
                data_dir_options.append(item_path)
    
    selected_dir = st.selectbox("Select Dataset Directory", data_dir_options, index=0)
    
    if not os.path.exists(selected_dir):
        st.warning(f"Directory `{selected_dir}` does not exist yet. Please run the data collection script first.")
        st.stop()
        
    # Find readable files
    # include typical format files + CIFAR batches + MNIST ubytes 
    data_files = (
        glob.glob(os.path.join(selected_dir, "*.npz")) + 
        glob.glob(os.path.join(selected_dir, "*.npy")) + 
        glob.glob(os.path.join(selected_dir, "*ubyte*")) +
        glob.glob(os.path.join(selected_dir, "data_batch_*")) +
        glob.glob(os.path.join(selected_dir, "test_batch*"))
    )
    
    # Filter out empty strings or unexpected results
    data_files = sorted(list(set(data_files)))
    
    if not data_files:
        st.warning(f"No parseable data files found in `{selected_dir}`.")
        st.stop()
        
    st.sidebar.subheader("Dataset Info")
    st.sidebar.text(f"Files found: {len(data_files)}")
    
    # Select episode/file
    selected_file = st.selectbox("Select File", data_files, format_func=lambda x: os.path.basename(x))
    
    # Load data dynamically
    @st.cache_data
    def load_data(filepath):
        basename = os.path.basename(filepath)
        try:
            if filepath.endswith('.npz'):
                data = np.load(filepath, allow_pickle=True)
                return dict(data)
            elif filepath.endswith('.npy'):
                return {"data": np.load(filepath, allow_pickle=True)}
            elif "ubyte" in basename:
                return {"mnist_data": read_idx(filepath)}
            elif "batch" in basename and not "meta" in basename:
                images, labels = load_cifar_batch(filepath)
                meta_file = os.path.join(os.path.dirname(filepath), "batches.meta")
                label_names = load_cifar_meta(meta_file) if os.path.exists(meta_file) else None
                return {"cifar_images": images, "cifar_labels": labels, "cifar_meta": label_names}
            else:
                 return {"error": "Unknown file format"}
        except Exception as e:
            return {"error": str(e)}

    data_dict = load_data(selected_file)
    
    if "error" in data_dict:
        st.error(f"Failed to load `{selected_file}`: {data_dict['error']}")
        st.stop()
        
    # Determine type of data and render it
    if "frames" in data_dict and "actions" in data_dict:
        # Standard FluidVLA Pick & Place format
        frames = data_dict.get("frames", np.array([]))
        proprios = data_dict.get("proprios", np.array([]))
        actions = data_dict.get("actions", np.array([]))
        reward = data_dict.get("reward", [0])[0]
        
        num_steps = len(actions)
        if num_steps == 0 or len(frames) == 0:
            st.warning("Empty episode data.")
            st.stop()
            
        frame_shape = frames.shape[1:] if len(frames.shape) > 1 else "Unknown"
        
        st.sidebar.text(f"Steps: {num_steps}")
        st.sidebar.text(f"Frame Shape: {frame_shape}")
        
        if reward <= 0:
            st.sidebar.error(f"Reward: {reward} ❌")
        else:
            st.sidebar.success(f"Reward: {reward} ✅")
            
        step_idx = st.slider("Step", min_value=0, max_value=max(0, num_steps - 1), value=0)
        st.subheader(f"Step {step_idx}/{num_steps - 1}")
        
        current_frames = frames[step_idx] if len(frames) > step_idx else None
        
        # Display frames
        if current_frames is not None and len(current_frames.shape) >= 3:
            st.markdown("**Visual Observation**")
            # Handle (3, T, H, W) or (T, 3, H, W) or (3, H, W)
            if len(current_frames.shape) == 4:
                T = current_frames.shape[1] if current_frames.shape[0] == 3 else current_frames.shape[0]
                cols = st.columns(min(T, 8))
                for t in range(min(T, 8)):
                    with cols[t]:
                        st.caption(f"Frame {t}")
                        if current_frames.shape[0] == 3: # (3, T, H, W)
                            img_np = current_frames[:, t, :, :].transpose(1, 2, 0)
                        else: # (T, 3, H, W)
                            img_np = current_frames[t].transpose(1, 2, 0)
                            
                        img_np = np.clip(img_np, 0.0, 1.0)
                        img_np = (img_np * 255).astype(np.uint8)
                        img = Image.fromarray(img_np)
                        if img.width < 256: img = img.resize((img.width * 4, img.height * 4), Image.NEAREST)
                        st.image(img, use_container_width=True)
            elif len(current_frames.shape) == 3: # (3, H, W)
                 img_np = current_frames.transpose(1, 2, 0)
                 img_np = np.clip(img_np, 0.0, 1.0)
                 img_np = (img_np * 255).astype(np.uint8)
                 img = Image.fromarray(img_np)
                 if img.width < 256: img = img.resize((img.width * 4, img.height * 4), Image.NEAREST)
                 st.image(img, width=400)
        
        col1, col2 = st.columns(2)
        with col1:
            if len(proprios) > step_idx:
                st.markdown("**Proprioception**")
                st.dataframe({"Value": proprios[step_idx]}, height=300)
        with col2:
            if len(actions) > step_idx:
                st.markdown("**Action**")
                st.dataframe({"Value": actions[step_idx]}, height=300)
                
        if len(actions.shape) >= 2 and actions.shape[1] >= 7:
            st.markdown("---")
            st.markdown("**Actions across episode**")
            fig, axes = plt.subplots(2, 1, figsize=(10, 6), sharex=True)
            axes[0].plot(actions[:, 0], label="x")
            axes[0].plot(actions[:, 1], label="y")
            axes[0].plot(actions[:, 2], label="z")
            axes[0].set_ylabel("Delta Position")
            axes[0].legend(); axes[0].grid(True, alpha=0.3)
            axes[0].axvline(x=step_idx, color='r', linestyle='--', alpha=0.5)
            
            axes[1].plot(actions[:, 3], label="rx")
            axes[1].plot(actions[:, 4], label="ry")
            axes[1].plot(actions[:, 5], label="rz")
            axes[1].plot(actions[:, 6], label="gripper", linestyle=':')
            axes[1].set_ylabel("Delta Rot & Gripper")
            axes[1].legend(); axes[1].grid(True, alpha=0.3)
            axes[1].axvline(x=step_idx, color='r', linestyle='--', alpha=0.5)
            st.pyplot(fig)
            

    elif "mnist_data" in data_dict:
        # MNIST / Fashion-MNIST format viewer
        data = data_dict["mnist_data"]
        st.subheader("MNIST (idx-ubyte) Viewer")
        st.text(f"Shape: {data.shape}")
        
        if len(data.shape) == 3: # Images (N, H, W)
            num_images = data.shape[0]
            step_idx = st.slider("Sample Index", min_value=0, max_value=num_images - 1, value=0)
            img_np = data[step_idx]
            img = Image.fromarray(img_np)
            img = img.resize((img.width * 10, img.height * 10), Image.NEAREST)
            st.image(img, caption=f"Sample {step_idx}")
        elif len(data.shape) == 1: # Labels (N,)
            st.dataframe({"Label": data}, height=400)
            

    elif "cifar_images" in data_dict:
        # CIFAR-10 viewer
        images = data_dict["cifar_images"]
        labels = data_dict["cifar_labels"]
        meta = data_dict["cifar_meta"]
        
        st.subheader("CIFAR-10 Batch Viewer")
        st.text(f"Images Shape: {images.shape}")
        
        num_images = images.shape[0]
        step_idx = st.slider("Sample Index", min_value=0, max_value=num_images - 1, value=0)
        
        img_np = images[step_idx]
        img_label = labels[step_idx]
        label_name = meta[img_label] if meta else f"Class {img_label}"
        
        img = Image.fromarray(img_np)
        img = img.resize((img.width * 8, img.height * 8), Image.NEAREST)
        st.image(img, caption=f"Label: {label_name}")


    else:
        # Generic Numpy Viewer
        st.info("This file doesn't follow the standard `frames/actions` format. Showing raw arrays.")
        for key, val in data_dict.items():
            if isinstance(val, np.ndarray):
                st.subheader(f"Array: `{key}`")
                st.text(f"Shape: {val.shape} | Dtype: {val.dtype} | Min: {val.min() if val.size > 0 else 'N/A'} | Max: {val.max() if val.size > 0 else 'N/A'}")
                if len(val.shape) <= 2:
                    st.dataframe(val)
                elif len(val.shape) == 3 and val.shape[-1] in [1, 3, 4]:
                    # Maybe image?
                    try:
                        st.image(val, caption=key, width=300)
                    except:
                        st.text("Could not render as image.")
            else:
                st.subheader(f"Value: `{key}`")
                st.write(val)


elif mode == "Model & Logs Explorer (.pt, .json)":
    st.header("🧠 Model Checkpoints & Logs")
    
    ckpt_dirs = ["./checkpoints/step0", "./checkpoints/step1", "./checkpoints/step2", "./checkpoints/step2_isaac"]
    if os.path.exists("./checkpoints"):
        for item in os.listdir("./checkpoints"):
            item_path = os.path.join(".", "checkpoints", item)
            if os.path.isdir(item_path) and item_path not in ckpt_dirs:
                ckpt_dirs.append(item_path)
                
    selected_ckpt_dir = st.selectbox("Select Checkpoint Directory", ckpt_dirs, index=0)
    
    if not os.path.exists(selected_ckpt_dir):
        st.warning(f"Directory `{selected_ckpt_dir}` does not exist.")
        st.stop()
        
    files = sorted(os.listdir(selected_ckpt_dir))
    pt_files = [f for f in files if f.endswith('.pt') or f.endswith('.pth')]
    json_files = [f for f in files if f.endswith('.json')]
    
    col_l, col_r = st.columns(2)
    
    with col_l:
        st.subheader("Training Logs (.json)")
        if not json_files:
            st.info("No JSON logs found in this directory.")
        else:
            selected_json = st.selectbox("Select Log File", json_files)
            json_path = os.path.join(selected_ckpt_dir, selected_json)
            try:
                with open(json_path, 'r') as f:
                    history = json.load(f)
                
                if isinstance(history, list) and len(history) > 0:
                    st.success(f"Loaded {len(history)} epochs.")
                    
                    # Extract numeric keys for plotting (ignore strictly structural or string keys)
                    keys = [k for k in history[0].keys() if isinstance(history[0][k], (int, float))]
                    keys = [k for k in keys if k != 'epoch']
                    
                    if keys:
                        # Allow user to pick exactly what to graph
                        st.markdown("**Metric Graph Visualization**")
                        # Default to the first two or more interesting metrics like Loss/MSE/Acc if they exist
                        default_keys = []
                        for possible_default in ["train_loss", "val_mse", "train_acc", "test_acc", "val_turb", "val_steps"]:
                            if possible_default in keys:
                                default_keys.append(possible_default)
                        if not default_keys: default_keys = keys[:2] if len(keys)>=2 else keys

                        selected_metrics = st.multiselect("Select Metrics to Plot", keys, default=default_keys)
                        
                        if selected_metrics:
                            epochs = [item.get('epoch', i+1) for i, item in enumerate(history)]
                            fig, ax = plt.subplots(figsize=(10, 5))
                            for m in selected_metrics:
                                vals = [item.get(m, 0) for item in history]
                                ax.plot(epochs, vals, label=m, marker='o' if len(epochs)<30 else None)
                            ax.set_xlabel("Epoch")
                            ax.set_ylabel("Value")
                            ax.legend()
                            ax.grid(True, alpha=0.3)
                            st.pyplot(fig)
                            
                    st.write("**Raw JSON Data (History Table):**")
                    st.dataframe(history)
                else:
                    st.json(history)
            except Exception as e:
                st.error(f"Error parsing JSON: {e}")

    with col_r:
        st.subheader("Model Checkpoints (.pt)")
        if not pt_files:
            st.info("No PyTorch checkpoints found in this directory.")
        else:
            selected_pt = st.selectbox("Select Checkpoint", pt_files)
            pt_path = os.path.join(selected_ckpt_dir, selected_pt)
            
            if st.button("Inspect Checkpoint Structure"):
                with st.spinner("Loading checkpoint (CPU mapped)..."):
                    try:
                        ckpt = torch.load(pt_path, map_location='cpu', weights_only=False)
                        st.success("Loaded successfully!")
                        
                        st.write("**Top-level Keys**")
                        st.write(list(ckpt.keys()) if isinstance(ckpt, dict) else type(ckpt))
                        
                        if isinstance(ckpt, dict):
                            if 'epoch' in ckpt:
                                st.metric("Saved at Epoch", ckpt['epoch'])
                            if 'val_mse' in ckpt:
                                st.metric("Validation MSE", f"{ckpt['val_mse']:.5f}")
                            if 'config' in ckpt:
                                st.write("**Model Config**")
                                st.json(ckpt['config'])
                            
                            if 'model' in ckpt or 'model_state_dict' in ckpt:
                                state_dict = ckpt.get('model', ckpt.get('model_state_dict'))
                                st.write(f"**Weights Info**: {len(state_dict)} layers found.")
                                
                                # Show tensor sizes
                                sizes = {k: list(v.shape) for k, v in state_dict.items()}
                                st.write("Tensor Shapes Breakdown:")
                                st.dataframe([{"Layer": k, "Shape": str(v)} for k, v in sizes.items()], height=400)
                                
                    except Exception as e:
                        st.error(f"Error loading checkpoint: {e}")
