
__all__ = ["GeneralMotionRetargeting"]


def __getattr__(name: str):

    if name == "GeneralMotionRetargeting":
        from .motion_retarget import GeneralMotionRetargeting

        return GeneralMotionRetargeting
    raise AttributeError(name)
