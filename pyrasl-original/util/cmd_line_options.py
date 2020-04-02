def parse_gpu_string(gpu_string):
    """
    Create a list of integer from a string of single digit integer characters.
    """
    l = []
    for char in gpu_string:
        l.append(int(char))
    return l

def parse_time_range_string(tr_string):
    """
    Parse the time point range with syntax: <start_time>:<end_time>.  Returns a tuple (t_start, t_end)
    """
    tlist = tr_string.split(':')
    if len(tlist) != 2:
        print 'Error parsing --time-range argument.  Be sure to use <start-time>:<end-time> syntax.'
        sys.exit(1)
    t_start = int(tlist[0])
    t_end   = int(tlist[1])
    return (t_start, t_end)

def parse_int_range_string(r_string):
    """
    Parse a range string with syntax: start:stop or start:step:stop 
    Returns a tuple (start, end, step)
    """
    rlist = r_string.split(':')
    if len(rlist) == 2:
        i_start = int(rlist[0])
        i_end   = int(rlist[1])
        i_step  = 1
    elif len(rlist) == 3:
        i_start = int(rlist[0])
        i_step  = int(rlist[1])
        i_end   = int(rlist[2])
    else:
        print 'Error parsing --time-range argument. Invalid syntax'
        sys.exit(1)
    return (i_start, i_end, i_step)

def parse_2D_ROI_string(ROI_string):
    """
    Parse the 2D ROI range with syntax: <start_x_ROI>:<end_x_ROI>:<start_y_ROI>:<end_y_ROI>. 
    Returns a tuple (start_x_ROI, end_x_ROI, start_y_ROI, end_y_ROI)
    """
    roi_list = ROI_string.split(':')
    if len(roi_list) != 4:
        print 'Error parsing --ROI argument.  Be sure to use <start_x_ROI>:<end_x_ROI>:<start_y_ROI>:<end_y_ROI> syntax.'
        sys.exit(1)
    start_x_ROI = int(roi_list[0])
    end_x_ROI = int(roi_list[1])
    start_y_ROI = int(roi_list[2])
    end_y_ROI = int(roi_list[3])
    return (start_x_ROI, end_x_ROI, start_y_ROI, end_y_ROI)
