<div align="center">
  
<h2> Depth Completion by Rescaling Monocular Depth Estimates via Compressed Sensing </h2>
<p>Daoxin Zhong, Jun Li, Yeshas Thadimari, Michael Chuah</p>
<p><a href="https://arxiv.org/">[arXiv]</a></p>

</div>

TLDR: Using compressed sensing to reconstruct scaling matrix to rescale estimated depth

---
### Data Format

---
### Docker

Build the docker image using the following command:

```
docker build \
    --ssh default=$SSH_AUTH_SOCK \
    -t rescale_compressed_sensing .
```

Run the docker image using the following command:

```
docker run -it -d \
  --restart unless-stopped \
  --gpus all \
  --network host \
  --shm-size 16g \
  --privileged \
  -v <path_of_scratch_data>:/scratchdata \
  -v /tmp/.X11-unix:/tmp/.X11-unix:rw \
  -e DISPLAY=$DISPLAY \
  -e QT_X11_NO_MITSHM=1 \
  --device /dev/ttyUSB0 \
  rescale_compressed_sensing
```
---
### Build


---

### Run


---

### Citation

```
@inproceedings{
}
```