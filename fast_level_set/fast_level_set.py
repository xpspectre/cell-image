def level_set(psi_, img, Tmax=500, U=3, V=1):
    """Level set algorithm



    Args:
        psi_: Previous region labeling map of frame k-1
        img (Numpy array of uint8): Input image
        Tmax (positive int): Max iterations of the main loop (default = 500)
        U (positive int): Update cycle iterations (default = 3)
        V (positive int): Regulation cycle iterations (default = 1)

    Returns:
        psi: Updated region labeling map of frame k

    References:
        Li, K., Miller, E. D., Chen, M., Kanade, T., Weiss, L. E., & Campbell, P. G. (2008). Cell population tracking
            and lineage construction with spatiotemporal context. Medical Image Analysis, 12(5), 546–566.
            http://doi.org/10.1016/j.media.2008.06.001
        Shi, Y., & Karl, W. C. (2005). Real-time Tracking Using Level Sets. IEEE Computer Society Conference on Computer
            Vision and Pattern Recognition, 2, 20–25. http://doi.org/10.1109/CVPR.2005.294

    """

    # Initialization

