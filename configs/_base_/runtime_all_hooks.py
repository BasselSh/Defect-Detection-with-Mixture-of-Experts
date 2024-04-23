default_scope = 'mmdet'
custom_imports = dict(imports=['wrappers'], allow_failed_imports=False)
default_hooks = dict(
    timer=dict(type='IterTimerHook'),
    logger=dict(type='LoggerHook', interval=50),
    param_scheduler=dict(type='ParamSchedulerHook'),
    #earlystop=dict(type='EarlyStoppingHook', monitor='bbox_mAP', rule='greater'),
    checkpoint=dict(type='CheckpointHook', interval=1, save_best='bbox_mAP', rule='greater'),
    clearml_vis=dict(type='ImageVisualizer',interval=20, iterations=10),
    sampler_seed=dict(type='DistSamplerSeedHook'),
    visualization=dict(type='DetVisualizationHook'),
    )

env_cfg = dict(
    cudnn_benchmark=False,
    mp_cfg=dict(mp_start_method='fork', opencv_num_threads=0),
    dist_cfg=dict(backend='nccl'),
)

vis_backends = [dict(type='LocalVisBackend')]

# vis_backends = [dict(type='ClearMLVisBackend',
#                     save_dir='clearml',
#                     init_kwargs=dict(project_name="Defect-Detection", task_name="simplecheck"))
#                 ]
visualizer = dict(
    type='DetLocalVisualizer', vis_backends=vis_backends, name='visualizer')
log_processor = dict(type='LogProcessor', window_size=50, by_epoch=True)

log_level = 'INFO'
load_from = None
resume = False
