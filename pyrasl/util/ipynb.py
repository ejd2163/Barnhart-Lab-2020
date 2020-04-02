def HtmlProgressBar():
    '''
    Create a HTML progress bar object below the current cell.  This code is 
    implemented using a closure, returning the an update() function object that,
    when called, updates the progress bar dynamically in the notebook.

    Example:

      pb = HtmlProgressBar()
      for i in range(100):
        pb(i)

    Note that you can call pb(i) as many times as you want... the Progress bar will 
    only update every 0.25 seconds, so there is no performance penalty for repeated
    calls.
    '''

    from IPython.display import HTML, Javascript, display
    import uuid
    divid = str(uuid.uuid4())
    tic = 0

    pb = HTML(
    """
    <div style="border: 1px solid black; width:500px">
      <div id="%s" style="background-color:blue; height: 10px; width:0%%">&nbsp;</div>
    </div> 
    """ % divid)
    display(pb)

    # We only want to update occasionally every quarter of a second at most!
    import time
    tic = [time.time()]  # Needs to be array, so that we can write to this scope from the closure.
   
    # Create a closure
    def update(pct, finished = False):
        if not finished:
            if time.time() - tic[0] > 0.25:
                display(Javascript("$('div#%s').width('%i%%')" % (divid, int(pct))))      
                tic[0] = time.time()
        else:
            display(Javascript("$('div#%s').width('100%%')" % (divid)))
        
    return update

def display_html5_video(url, width = 640):
    '''
    Displays an HTML 5 video in the iPython notebook.  The URL must point at a video encoded
    with a codec supported by your web browser.  The file must also reside on a web server, 
    rather than the local filesystem, since security restrictions in the browser prevent Loading
    arbitrary files on disk.
    '''

    from IPython.display import HTML, display
    video_tag = '<center><video controls alt="Calcium Activity Movie" width=%d src="%s"></center>' % (width, url)
    display( HTML(data=video_tag) ) 
