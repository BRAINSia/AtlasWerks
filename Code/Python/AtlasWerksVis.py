from AtlasWerks.DataTypes import *

import vtk

# import matplotlib

# matplotlib.use('QtAgg')
# # the following is in case you want matplotlib to do some vtk stuff (like 3D quiver)
# #matplotlib.use('Agg')

# import matplotlib.pyplot
# matplotlib.pyplot.ion()

import numpy
import matplotlib.pylab

from scitools.std import *
from scitools.easyviz.vtk_ import *

DIR_X, DIR_Y, DIR_Z = range(3)

# the following is for volume rendering code from http://www.siafoo.net/snippet/314
'''
A quick and dirty VTK based viewer for volume data.
'''

import sys, wx

try:
    from TransferFunction import TransferFunctionWidget
except ImportError:
    print 'You need to get the TransferFunction from: "http://www.siafoo.net/snippet/122"'
    exit

try:
    from wxVTKRenderWindowInteractor import wxVTKRenderWindowInteractor
except ImportError:
    print 'Warning: If you are getting flickering get wxVTKRenderWindowInteractor from: "http://www.siafoo.net/snippet/312"'
    from vtk.wx.wxVTKRenderWindowInteractor import wxVTKRenderWindowInteractor

try:       
    from vtkImageImportFromArray import vtkImageImportFromArray
except ImportError:
    print 'You need to get the updated vtkImageImportFromArray from: "http://www.siafoo.net/snippet/313" '
    exit

class TransferGraph(wx.Dialog):

    def __init__(self, parent, id=wx.ID_ANY, title="", pos=wx.DefaultPosition,
                 size=wx.DefaultSize, style=wx.DEFAULT_FRAME_STYLE):

        wx.Dialog.__init__(self, parent, id, title, pos, size, style)
        self.mainPanel = wx.Panel(self, -1)

        # Create some CustomCheckBoxes
        self.t_function = TransferFunctionWidget(self.mainPanel, -1, "", size=wx.Size(300, 150))
       
        # Layout the items with sizers
        mainSizer = wx.BoxSizer(wx.VERTICAL)
        mainSizer.Add(self.mainPanel, 1, wx.EXPAND)
        
        self.SetSizer(mainSizer)
        mainSizer.Layout()
        
    def get_function(self):
        return self.t_function

class VolumeRenderHeadVTK(wxVTKRenderWindowInteractor):
    def __init__(self, parent, data_set):
        wxVTKRenderWindowInteractor.__init__(self, parent, -1, size=parent.GetSize())
        
        ren = vtk.vtkRenderer()
        self.GetRenderWindow().AddRenderer(ren)

        # prob should allow a string as data_set also, but for now assume it's a python or numpy array
        #img = self.LoadVolumeData(data_set) # load from filename
        img = vtkImageImportFromArray()
        img.SetArray(data_set)

        pix_diag = 5.0

        # volMapper = vtk.vtkVolumeTextureMapper3D()

        volMapper = vtk.vtkVolumeRayCastMapper()
        compositeFunction = vtk.vtkVolumeRayCastCompositeFunction()
        compositeFunction.SetCompositeMethodToInterpolateFirst()
        volMapper.SetVolumeRayCastFunction(compositeFunction)

        volMapper.SetSampleDistance(pix_diag / 5.0)
        volMapper.SetInputConnection(img.GetOutputPort())

        # Transfer Functions
        self.opacity_tf = vtk.vtkPiecewiseFunction()
        self.color_tf = vtk.vtkColorTransferFunction()

        # The property describes how the data will look
        self.volProperty = volProperty = vtk.vtkVolumeProperty()
        volProperty.SetColor(self.color_tf)
        volProperty.SetScalarOpacity(self.opacity_tf)
        volProperty.ShadeOn()
        volProperty.SetInterpolationTypeToLinear()
        volProperty.SetScalarOpacityUnitDistance(pix_diag)

        vol = vtk.vtkVolume()
        vol.SetMapper(volMapper)
        vol.SetProperty(volProperty)
        
        ren.AddVolume(vol)
        
        boxWidget = vtk.vtkBoxWidget()
        boxWidget.SetInteractor(self)
        boxWidget.SetPlaceFactor(1.0)
        
        # The implicit function vtkPlanes is used in conjunction with the
        # volume ray cast mapper to limit which portion of the volume is
        # volume rendered.
        planes = vtk.vtkPlanes()
        def ClipVolumeRender(obj, event):
            obj.GetPlanes(planes)
            volMapper.SetClippingPlanes(planes)
        
        # Place the interactor initially. The output of the reader is used to
        # place the box widget.
        boxWidget.SetInput(img.GetOutput())
        boxWidget.PlaceWidget()
        boxWidget.InsideOutOn()
        boxWidget.AddObserver("InteractionEvent", ClipVolumeRender)

        outline = vtk.vtkOutlineFilter()
        outline.SetInputConnection(img.GetOutputPort())
        outlineMapper = vtk.vtkPolyDataMapper()
        outlineMapper.SetInputConnection(outline.GetOutputPort())
        outlineActor = vtk.vtkActor()
        outlineActor.SetMapper(outlineMapper)

        outlineProperty = boxWidget.GetOutlineProperty()
        outlineProperty.SetRepresentationToWireframe()
        outlineProperty.SetAmbient(1.0)
        outlineProperty.SetAmbientColor(1, 1, 1)
        outlineProperty.SetLineWidth(3)
        
        selectedOutlineProperty = boxWidget.GetSelectedOutlineProperty()
        selectedOutlineProperty.SetRepresentationToWireframe()
        selectedOutlineProperty.SetAmbient(1.0)
        selectedOutlineProperty.SetAmbientColor(1, 0, 0)
        selectedOutlineProperty.SetLineWidth(3)

        ren.AddActor(outlineActor)
        
        style = vtk.vtkInteractorStyleTrackballCamera()
        self.SetInteractorStyle(style)
        self.t_graph = TransferGraph(self)
        self.SetFocus()
        
        # So we know when new values are added / changed on the tgraph
        self.t_graph.Connect(-1, -1, wx.wxEVT_COMMAND_SLIDER_UPDATED, 
                             self.OnTGraphUpdate)
       
        self.OnTGraphUpdate(None)
        
        self.lighting = True

        # This is the transfer graph
        self.t_graph.Show()

    def OnTGraphUpdate(self, event):

        self.color_tf.RemoveAllPoints()
        self.opacity_tf.RemoveAllPoints()
        
        for p in self.t_graph.t_function.points:
            rgba = p.get_rgba()
            self.color_tf.AddRGBPoint(p.value, rgba[0]/255.0, 
                                 rgba[1]/255.0, rgba[2]/255.0)

            self.opacity_tf.AddPoint(p.value, rgba[3])

        self.Refresh()

    def LoadVolumeData(self, data_set):

        if data_set.endswith('nii') or data_set.endswith('nii.gz'):
            try:
                from nifti import NiftiImage
            except ImportError:
                print "Apparently you don't have PyNIfTI installed, see http://www.siafoo.net/snippet/310 for instructions"
                exit(1)
            nim = NiftiImage(data_set)
            img_data = nim.data
        elif data_set.endswith('hdr'):
            # Use the header to figure out the shape of the data
            # then load the raw data and reshape the array 
            shape = [int(x) for x in open(data_set).readline().split()]
            img_data = numpy.frombuffer(open(data_set.replace('.hdr', '.dat'), 'rb').read(),
                                        numpy.uint8)\
                        .reshape((shape[2], shape[1], shape[0]))
                        
        img = vtkImageImportFromArray()
        img.SetArray(img_data)
        
        return img

    def OnKeyDown(self, event):
        # Yes, I did just hack this... maybe one day when things work
        # the way the docs claim... hacks won't be necessary.
        
        key_code = event.GetKeyCode()
        
        if key_code == ord('l'):
            self.lighting = not self.lighting
            print 'Lighting', self.lighting 
            self.volProperty.SetShade(self.lighting)
            self.Refresh()
        else:
            wxVTKRenderWindowInteractor.OnKeyDown(self, event)


def VolumeRender(imgAW, title='Volume Rendering with VTK', winsizex=600, winsizey=600):
    """
    Volume render a fImage using code from http://www.siafoo.net/snippet/314
    """
    app = wx.App()
    frame = wx.Frame(app, -1, title, wx.DefaultPosition, wx.Size(winsizex,winsizey))

    Iarr = numpy.array(imgAW.tolist())
    canvas = VolumeRenderHeadVTK(frame, Iarr)

    frame.Show()
    app.MainLoop()

def Quiver3D(vf):
    """
    Creates a 3D quiver plot of an fVectorField

    Note this uses the VTK backend of scitools, so you need python-vtk
    and scitools (as well as numpy) installed
    """

    # as always convert to a list or numpy array first
    l = numpy.array(vf.tolist())

    nx,ny,nz,nv = shape(l)
    sx = round(nx/16)
    sy = round(ny/16)
    sz = round(nz/16)

    U = numpy.squeeze(l[::sx,::sy,::sz,0])
    V = numpy.squeeze(l[::sx,::sy,::sz,1])
    W = numpy.squeeze(l[::sx,::sy,::sz,2])

    x = range(size(U,0))
    y = range(size(U,1))
    z = range(size(U,2))
    X,Y,Z = ndgrid(x,y,z)
    quiver3(X,Y,Z,U,V,W,0.001)

def QuiverSlice(vf, sliceDir, sliceNum, scale=1):
    """
    Extracts a slice from a vector field and creates a quiver plot of it.
    """
    sl = vf.getSlice(sliceDir, int(sliceNum)).flatten(sliceDir)

    # get U and V, just component lists for vf
    s = numpy.array(sl.tolist())
    U = numpy.squeeze(s[:,:,0])
    V = numpy.squeeze(s[:,:,1])

    # only take every nth arrow so we don't get a superhuge quiver plot
    nx = max(round(sl.getSizeX()/32),1)
    ny = max(round(sl.getSizeY()/32),1)

    Q = matplotlib.pylab.quiver(U[::nx,::ny],V[::nx,::ny], scale)

def ThreePaneView(I):
    """
    Pass in an fImage, fArray3D, python list, or numpy array and we'll
    show an interactive threepane view.
    """
    print("Threepane view not implemented yet, just an idea.")

def ThreePaneView4D(I0,vfwd,vrev,baseamp,deltaamp):
    """
    Pass in an fImage, fArray3D, python list, or numpy array as a base
    image and a list of vector fields, base amplitude and delta
    amplitude and we'll show a three-pane view with a slider on the
    bottom for visualizing 4D flows of images
    """
    print("4D threepane view not implemented yet, just an idea.")

def ThreePaneView4DFrames(It):
    """
    Pass in a series of fImages, fArray3Ds, python lists, or numpy
    array and we'll show a three-pane view with a slider on the bottom
    for visualizing 4D datasets as 3D frames
    """
    print("4D frame-based threepane view not implemented yet, just an idea.")
