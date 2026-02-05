from transformers import Sam3VideoModel, Sam3VideoProcessor
from accelerate import Accelerator
import torch
import av
import imageio
import numpy as np
from PIL import Image, ImageFilter
import imageio.v3 as iio
from transformers.video_utils import load_video
import logging
import oci
from oci.auth import signers
import os
from huggingface_hub import login

logging.info("packages loaded")

login(token=os.getenv['HF_TOKEN'], add_to_git_credential=False) # SAM3 requires HF login to download weights. Either us OS variable or OCI Vault to store secret

logging.info("logged into huggingface")

device = Accelerator().device

logging.info(str("using device type "+str(device.type)))

model = Sam3VideoModel.from_pretrained("facebook/sam3").to(device, dtype=torch.bfloat16)

processor = Sam3VideoProcessor.from_pretrained("facebook/sam3")

logging.info("model weights loaded")

signer = signers.get_resource_principals_signer()
object_storage_client = oci.object_storage.ObjectStorageClient({}, signer=signer)

NAMESPACE = object_storage_client.get_namespace().data
BUCKET = "sam3"
SRC_NAME = "simple_video.mp4"
SRC_OBJECT = "inputs/"+SRC_NAME
DST_OBJECT = "outputs/blurred_"+SRC_NAME  # new object/key for output
LOCAL_VIDEO = "/tmp/input_video.mp4"
LOCAL_OUT = "/tmp/output_video.mp4"

#Download the input video from Object Storage (using resource princiapl auth)
print("Downloading video from Object Storage...")
with open(LOCAL_VIDEO, "wb") as f:
    get_obj = object_storage_client.get_object(NAMESPACE, BUCKET, SRC_OBJECT)
    for chunk in get_obj.data.raw.stream(1024*1024, decode_content=False):
        f.write(chunk)
print("Downloaded video to:", LOCAL_VIDEO)
logging.info("video downloaded from object storage")

# load the video downloaded to tmp
video_frames, _ = load_video(LOCAL_VIDEO)

logging.info("video frames loaded")

num_frames = "number of frames"+str(len(video_frames))
logging.info(str(num_frames))

inference_session = processor.init_video_session(
    video=video_frames,
    inference_device=device,
    processing_device="cpu",
    video_storage_device="cpu",
    dtype=torch.bfloat16,
)

text = "face"
inference_session = processor.add_text_prompt(
    inference_session=inference_session,
    text=text,
)

outputs_per_frame = {}
logging.info("text prompt added")
for model_outputs in model.propagate_in_video_iterator(inference_session=inference_session, max_frame_num_to_track=len(video_frames)):
    processed_outputs = processor.postprocess_outputs(inference_session, model_outputs)
    outputs_per_frame[model_outputs.frame_idx] = processed_outputs

print(f"Processed {len(outputs_per_frame)} frames")

# Set output file and FPS
out_path = LOCAL_OUT
fps = 30  

frames_to_write = []
logging.info("adding blur to source video")
for idx, frame in enumerate(video_frames):
    frame_outputs = outputs_per_frame.get(idx)
    # Always convert to PIL for processing
    frame_pil = frame if isinstance(frame, Image.Image) else Image.fromarray(frame)
    
    if not frame_outputs:
        frames_to_write.append(np.array(frame_pil))
        continue

    masks = frame_outputs['masks']
    if isinstance(masks, torch.Tensor):
        masks = masks.cpu().numpy()
    combined_mask = np.any(masks, axis=0)
    blurred = frame_pil.filter(ImageFilter.GaussianBlur(radius=40))
    mask_img = Image.fromarray((combined_mask * 255).astype('uint8')).convert('L')
    result = Image.composite(blurred, frame_pil, mask_img)
    frames_to_write.append(np.array(result))

iio.imwrite(out_path, frames_to_write, fps=fps,codec='libx264')
print(f"Saved anonymized video to {out_path}")

print("Uploading processed video to Object Storage...")
# upload output video back to object storage
with open(LOCAL_OUT, "rb") as f:
    object_storage_client.put_object(NAMESPACE, BUCKET, DST_OBJECT, f)
print("Uploaded anonymized video as:", DST_OBJECT)
