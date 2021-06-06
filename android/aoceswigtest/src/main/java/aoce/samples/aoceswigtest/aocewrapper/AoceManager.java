package aoce.samples.aoceswigtest.aocewrapper;

import java.util.List;

import aoce.android.library.xswig.*;

public class AoceManager extends IVideoDeviceObserver {
    private IPipeGraph pipeGraph = null;
    private LayerFactory layerFactory = null;
    private IInputLayer inputLayer = null;
    private IOutputLayer outputLayer = null;
    private IYUVLayer yuv2RGBALayer = null;
    private ITransposeLayer transposeLayer = null;
    private IFlipLayer flipLayer = null;
    private IReSizeLayer reSizeLayer = null;
    private IBaseLayer extraLayer = null;
    private IVideoDevice videoDevice = null;
    private VideoFormat videoFormat = null;
    private GLOutGpuTex gpuTex = new GLOutGpuTex();

    public void initGraph(){
        pipeGraph = AoceWrapper.getPipeGraphFactory(GpuType.vulkan).createGraph();
        layerFactory = AoceWrapper.getLayerFactory(GpuType.vulkan);
        inputLayer = layerFactory.createInput();
        outputLayer = layerFactory.createOutput();
        yuv2RGBALayer = layerFactory.createYUV2RGBA();
        transposeLayer = layerFactory.createTranspose();
        flipLayer = layerFactory.createFlip();
        reSizeLayer = layerFactory.createSize();

        TransposeParamet tp = transposeLayer.getParamet();
        tp.setBFlipX(1);
        tp.setBFlipY(1);
        transposeLayer.updateParamet(tp);
    }

    public void openCamera(){openCamera(false);}

    public void openCamera(boolean bFront){
        if(videoDevice != null && videoDevice.bOpen()){
            videoDevice.close();
        }
        int deviceCount = AoceWrapper.getVideoManager(CameraType.and_camera2).getDeviceCount();
        for(int i=0;i<deviceCount;i++){
            videoDevice = AoceWrapper.getVideoManager(CameraType.and_camera2).getDevice(i);
            if(videoDevice.back() != bFront){
                break;
            }
        }
        int formatIndex = videoDevice.findFormatIndex(1280,720);
        if(formatIndex < 0){
            formatIndex = 0;
        }
        videoDevice.setFormat(formatIndex);
        videoDevice.open();

        videoFormat = videoDevice.getSelectFormat();
        videoDevice.setObserver(this);
    }

    public void closeCamera(){
        if(videoDevice != null){
            videoDevice.close();
        }
    }

    public void initLayers(List<IBaseLayer> baseLayers,boolean bAutoIn){
        pipeGraph.clear();
        extraLayer = pipeGraph.addNode(inputLayer).addNode(yuv2RGBALayer);
        if(baseLayers != null) {
            for (IBaseLayer baseLayer : baseLayers) {
                extraLayer = extraLayer.addNode(baseLayer);
            }
            if (bAutoIn) {
                yuv2RGBALayer.getLayer().addLine(extraLayer, 0, 1);
            }
        }
        extraLayer.addNode(transposeLayer).addNode(outputLayer);
    }

    @Override
    public void onVideoFrame(VideoFrame frame){
        if(AoceWrapper.getYuvIndex(frame.getVideoType()) < 0){
            yuv2RGBALayer.getLayer().setVisable(false);
        }else if(videoFormat != null){
            if(yuv2RGBALayer.getParamet().getType() != frame.getVideoType()){
                yuv2RGBALayer.getLayer().setVisable(true);
                YUVParamet yp = yuv2RGBALayer.getParamet();
                yp.setType(frame.getVideoType());
                yuv2RGBALayer.updateParamet(yp);
            }
        }
        inputLayer.inputCpuData(frame);
        pipeGraph.run();
    }

    public void showGL(int textureId, int width, int height){
        gpuTex.setImage(textureId);
        gpuTex.setWidth(width);
        gpuTex.setHeight(height);
        outputLayer.outGLGpuTex(gpuTex);
    }
}
