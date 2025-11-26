CUDA_VISIBLE_DEVICES=0 \
python eval_seg.py --config ./configs/cfg_context59.py --work-dir YOUR_WORK_DIR &
CUDA_VISIBLE_DEVICES=1 \
python eval_seg.py --config ./configs/cfg_context60.py --work-dir YOUR_WORK_DIR &
CUDA_VISIBLE_DEVICES=2 \
python eval_seg.py --config ./configs/cfg_coco_object.py --work-dir YOUR_WORK_DIR & 
CUDA_VISIBLE_DEVICES=3 \
python eval_seg.py --config ./configs/cfg_coco_stuff164k.py --work-dir YOUR_WORK_DIR &
CUDA_VISIBLE_DEVICES=4 \
python eval_seg.py --config ./configs/cfg_ade20k.py --work-dir YOUR_WORK_DIR &
CUDA_VISIBLE_DEVICES=5 \
python eval_seg.py --config ./configs/cfg_voc20.py --work-dir YOUR_WORK_DIR &
CUDA_VISIBLE_DEVICES=6 \
python eval_seg.py --config ./configs/cfg_voc21.py --work-dir YOUR_WORK_DIR &
CUDA_VISIBLE_DEVICES=7 \
python eval_seg.py --config ./configs/cfg_city_scapes.py --work-dir YOUR_WORK_DIR &
wait