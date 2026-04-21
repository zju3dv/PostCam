mkdir -p checkpoints
cd checkpoints

GIT_LFS_SKIP_SMUDGE=1 git clone https://huggingface.co/CCQAQ/PostCam
cd PostCam && git lfs pull && cd ..

GIT_LFS_SKIP_SMUDGE=1 git clone https://huggingface.co/Wan-AI/Wan2.1-T2V-1.3B
cd Wan2.1-T2V-1.3B && git lfs pull && cd ..

GIT_LFS_SKIP_SMUDGE=1 git clone https://huggingface.co/depth-anything/DA3NESTED-GIANT-LARGE DA3
cd DA3 && git lfs pull && cd ..

GIT_LFS_SKIP_SMUDGE=1 git clone https://huggingface.co/microsoft/Florence-2-large
cd Florence-2-large && git lfs pull && cd ..


echo "All models have been downloaded and organized into the checkpoints/ directory."