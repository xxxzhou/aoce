package aoce.samples.aoceswigtest;

import java.lang.reflect.Method;

import aoce.android.library.xswig.ILMetadata;
import aoce.android.library.xswig.LayerMetadataType;

public class LayerMetaTag {
    public LayerMetadataType metadataType = LayerMetadataType.other;
    public ILMetadata layerMeta = null;
    public Object layerObj = null;
    public Object parametObj = null;
    public Method setMethod = null;
}
