"""Viewer exports load lazily so dashboard telemetry needs no renderer SDK."""
from importlib import import_module

_MODULES={
    'drawers':('MujocoSceneDrawer',),
    'factory':('add_viewer_backend_args','create_human_frame_viewer','create_retargeting_viewer'),
    'human_frame_viewer':('HumanFrameMujocoViewer','create_overlay_world_model','human_frame_positions'),
    'layers':('DEFAULT_HAND_POINT_SIZE','DEFAULT_HAND_LINE_WIDTH','DEFAULT_HAND_TRANSITION_LINE_WIDTH',
        'DEFAULT_HAND_WRIST_LINE_WIDTH','DEFAULT_HUMAN_POINT_SIZE','DEFAULT_HUMAN_LINE_WIDTH',
        'CommandVelocityArrowLayer','HumanFrameAxesLayer','HumanSkeletonLayer','VisualizationLayer'),
    'robot_support':('RobotSupportBaseLayer','RobotSupportState','compute_robot_support_state'),
    'retargeting_viewer':('RetargetingMujocoViewer',),
    'viser_viewer':('HumanFrameViserViewer','RetargetingViserViewer'),
}
__all__=[name for names in _MODULES.values() for name in names]


def __getattr__(name):
    for module,names in _MODULES.items():
        if name in names:
            value=getattr(import_module('.'+module,__name__),name)
            globals()[name]=value
            return value
    raise AttributeError(name)
