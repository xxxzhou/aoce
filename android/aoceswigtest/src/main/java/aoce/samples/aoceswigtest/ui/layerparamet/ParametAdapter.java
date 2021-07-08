package aoce.samples.aoceswigtest.ui.layerparamet;

import android.content.Context;
import android.view.View;
import android.view.ViewGroup;
import android.widget.CheckBox;
import android.widget.CompoundButton;
import android.widget.RadioGroup;
import android.widget.SeekBar;
import android.widget.TextView;

import java.lang.reflect.InvocationTargetException;
import java.lang.reflect.Method;
import java.text.DecimalFormat;
import java.util.ArrayList;

import androidx.annotation.NonNull;
import androidx.recyclerview.widget.RecyclerView;
import aoce.android.library.xswig.*;
import aoce.samples.aoceswigtest.DataManager;
import aoce.samples.aoceswigtest.LayerMetaTag;
import aoce.samples.aoceswigtest.R;
import aoce.samples.aoceswigtest.ui.layergroup.LayerAdapter;
import butterknife.BindView;
import butterknife.ButterKnife;

public class ParametAdapter extends RecyclerView.Adapter<RecyclerView.ViewHolder> {
    private Context context = null;
    private DataManager.LayerItem layerItem = null;
    private ILMetadata metadata = null;
    private ArrayList<LayerMetaTag> childs = new ArrayList<>();

    private void initMetadata(ILMetadata clMeta, Object obj, Object parentObj, String parametName) {
        try {
            if (clMeta.getLayerType() == LayerMetadataType.agroup) {
                ILGroupMetadata groupMetadata = AoceWrapper.getLGroupMetadata(clMeta);
                int lcount = groupMetadata.getCount();
                for (int i = 0; i < lcount; i++) {
                    ILMetadata ccMeta = groupMetadata.getLMetadata(i);
                    String cparametName = ccMeta.getParametName();
                    String jparametName = cparametName.substring(0, 1).toUpperCase() + cparametName.substring(1);
                    String methodStr = "get" + jparametName;
                    Method getMethod = null;
                    getMethod = obj.getClass().getMethod(methodStr);
                    Object childObj = getMethod.invoke(obj);
                    initMetadata(ccMeta, childObj, obj, "set" + jparametName);
                }
            } else {
                LayerMetaTag metaTag = new LayerMetaTag();
                metaTag.metadataType = clMeta.getLayerType();
                metaTag.parametObj = obj;
                metaTag.layerObj = parentObj;
                // 把obj的类型拆箱
                metaTag.setMethod = parentObj.getClass().getMethod(parametName, DataManager.getPrimitive(obj.getClass()));
                metaTag.layerMeta = clMeta;
                childs.add(metaTag);
            }
        } catch (NoSuchMethodException e) {
            e.printStackTrace();
        } catch (IllegalAccessException e) {
            e.printStackTrace();
        } catch (InvocationTargetException e) {
            e.printStackTrace();
        }
    }

    public ParametAdapter(Context context, int groupIndex, int layerIndex) {
        this.context = context;
        this.layerItem = DataManager.getInstance().getIndex(groupIndex).layers.get(layerIndex);
        if (layerItem.metadata != null) {
            this.metadata = AoceWrapper.getLayerMetadata(layerItem.metadata);
            if (layerItem.layers.size() > 0) {
                Object baseLayer = layerItem.getLayer();
                if (baseLayer instanceof ILayer) {
                    try {
                        initMetadata(metadata, layerItem.getParamet(), layerItem.getLayer(), "updateParamet");
                    } catch (NoSuchMethodException e) {
                        e.printStackTrace();
                    } catch (InvocationTargetException e) {
                        e.printStackTrace();
                    } catch (IllegalAccessException e) {
                        e.printStackTrace();
                    }
                }
            }
        }
    }

    @NonNull
    @Override
    public RecyclerView.ViewHolder onCreateViewHolder(@NonNull ViewGroup parent, int viewType) {
        LayerMetadataType metadataType = LayerMetadataType.swigToEnum(viewType);
        if (metadataType == LayerMetadataType.abool) {
            View view = View.inflate(parent.getContext(), R.layout.item_paramet_bool, null);
            return new PamaretBoolViewHolder(view);
        } else if (metadataType == LayerMetadataType.aint) {
            View view = View.inflate(parent.getContext(), R.layout.item_paramet_int, null);
            return new PamaretIntViewHolder(view);
        } else if (metadataType == LayerMetadataType.afloat) {
            View view = View.inflate(parent.getContext(), R.layout.item_paramet_float, null);
            return new PamaretFloatViewHolder(view);
        }
        return null;
    }

    @Override
    public int getItemViewType(int position) {
        ILMetadata metadata = (ILMetadata) childs.get(position).layerMeta;
        return metadata.getLayerType().swigValue();
    }

    @Override
    public void onBindViewHolder(@NonNull RecyclerView.ViewHolder holder, int position) {
        // LayerMetadataType metadataType = LayerMetadataType.swigToEnum(getItemViewType(position));
        BasePamaretViewHolder basePamaretViewHolder = (BasePamaretViewHolder) holder;
        basePamaretViewHolder.bintItem(childs.get(position), new PamaretChangeBack() {
            @Override
            public void onDataChange(Object object) {
                if (object != layerItem.getLayer()) {
                    try {
                        layerItem.updateParamet();
                    } catch (NoSuchMethodException e) {
                        e.printStackTrace();
                    } catch (InvocationTargetException e) {
                        e.printStackTrace();
                    } catch (IllegalAccessException e) {
                        e.printStackTrace();
                    }
                }
            }
        });
    }

    @Override
    public int getItemCount() {
        return childs.size();
    }

    public interface PamaretChangeBack {
        public void onDataChange(Object object);
    }

    public abstract class BasePamaretViewHolder extends RecyclerView.ViewHolder {
        protected LayerMetaTag metaTag = null;
        protected PamaretChangeBack dataChange = null;

        public BasePamaretViewHolder(@NonNull View itemView) {
            super(itemView);
        }

        public void bintItem(LayerMetaTag metaTag, PamaretChangeBack onChangeBack) {
            this.metaTag = metaTag;
            this.dataChange = onChangeBack;
            onBindItem();
            setValue();
        }

        public abstract void onBindItem();

        public abstract void setValue();
    }

    class PamaretBoolViewHolder extends BasePamaretViewHolder {
        @BindView(R.id.checkBox2)
        CheckBox checkBox;
        private ILBoolMetadata boolMetadata = null;

        PamaretBoolViewHolder(View view) {
            super(view);
            ButterKnife.bind(this, view);
        }

        @Override
        public void onBindItem() {
            boolMetadata = AoceWrapper.getLBoolMetadata(metaTag.layerMeta);
            checkBox.setText(metaTag.layerMeta.getText());
            checkBox.setOnCheckedChangeListener(new CheckBox.OnCheckedChangeListener() {
                @Override
                public void onCheckedChanged(CompoundButton buttonView, boolean isChecked) {
                    try {
                        if (DataManager.getPrimitive(metaTag.parametObj.getClass()) == boolean.class) {
                            metaTag.setMethod.invoke(metaTag.layerObj, isChecked);
                        } else {
                            metaTag.setMethod.invoke(metaTag.layerObj, isChecked ? 1 : 0);
                        }
                        dataChange.onDataChange(metaTag.layerObj);
                    } catch (IllegalAccessException e) {
                        e.printStackTrace();
                    } catch (InvocationTargetException e) {
                        e.printStackTrace();
                    }
                }
            });
        }

        @Override
        public void setValue() {
            boolean value = false;
            if (DataManager.getPrimitive(metaTag.parametObj.getClass()) == boolean.class) {
                value = (boolean) metaTag.parametObj;
            } else {
                value = (int) metaTag.parametObj != 0;
            }
            checkBox.setChecked(value);
        }
    }

    class PamaretIntViewHolder extends BasePamaretViewHolder {
        @BindView(R.id.textView9)
        TextView textView;
        @BindView(R.id.seekBar7)
        SeekBar seekBar;
        @BindView(R.id.textView10)
        TextView vauleView;
        private ILIntMetadata intMetadata = null;

        PamaretIntViewHolder(View view) {
            super(view);
            ButterKnife.bind(this, view);
        }

        @Override
        public void onBindItem() {
            intMetadata = AoceWrapper.getLIntMetadata(metaTag.layerMeta);
            textView.setText(metaTag.layerMeta.getText());
            seekBar.setOnSeekBarChangeListener(new SeekBar.OnSeekBarChangeListener() {
                @Override
                public void onProgressChanged(SeekBar seekBar, int progress, boolean fromUser) {
                    try {
                        if (DataManager.getPrimitive(metaTag.parametObj.getClass()) == long.class) {
                            metaTag.setMethod.invoke(metaTag.layerObj, (long) progress);
                        } else {
                            metaTag.setMethod.invoke(metaTag.layerObj, progress);
                        }
                        dataChange.onDataChange(metaTag.layerObj);
                    } catch (IllegalAccessException e) {
                        e.printStackTrace();
                    } catch (InvocationTargetException e) {
                        e.printStackTrace();
                    }
                    vauleView.setText(String.valueOf(progress));
                }

                @Override
                public void onStartTrackingTouch(SeekBar seekBar) {

                }

                @Override
                public void onStopTrackingTouch(SeekBar seekBar) {

                }
            });
        }

        @Override
        public void setValue() {
            seekBar.setMin(intMetadata.getMinValue());
            seekBar.setMax(intMetadata.getMaxValue());
            if(DataManager.getPrimitive(metaTag.parametObj.getClass()) == long.class){
                long value = (long) metaTag.parametObj;
                seekBar.setProgress((int) value);
            }else {
                seekBar.setProgress((int) metaTag.parametObj);
            }
        }
    }

    class PamaretFloatViewHolder extends BasePamaretViewHolder {
        @BindView(R.id.textView7)
        TextView textView;
        @BindView(R.id.seekBar6)
        SeekBar seekBar;
        @BindView(R.id.textView8)
        TextView vauleView;
        private ILFloatMetadata floatMetadata = null;

        PamaretFloatViewHolder(View view) {
            super(view);
            ButterKnife.bind(this, view);
        }

        @Override
        public void onBindItem() {
            floatMetadata = AoceWrapper.getLFloatMetadata(metaTag.layerMeta);
            textView.setText(metaTag.layerMeta.getText());
            seekBar.setOnSeekBarChangeListener(new SeekBar.OnSeekBarChangeListener() {
                @Override
                public void onProgressChanged(SeekBar seekBar, int progress, boolean fromUser) {
                    float value = (float) progress / 1000;
                    value = value * (floatMetadata.getMaxValue() - floatMetadata.getMinValue()) + floatMetadata.getMinValue();
                    DecimalFormat decimalFormat = new DecimalFormat("0.00");
                    vauleView.setText(decimalFormat.format(value));
                    try {
                        metaTag.setMethod.invoke(metaTag.layerObj, value);
                        dataChange.onDataChange(metaTag.layerObj);
                    } catch (IllegalAccessException e) {
                        e.printStackTrace();
                    } catch (InvocationTargetException e) {
                        e.printStackTrace();
                    }
                }

                @Override
                public void onStartTrackingTouch(SeekBar seekBar) {

                }

                @Override
                public void onStopTrackingTouch(SeekBar seekBar) {

                }
            });
        }

        @Override
        public void setValue() {
            float range = floatMetadata.getMaxValue() - floatMetadata.getMinValue();
            float cv = (float) metaTag.parametObj - floatMetadata.getMinValue();
            seekBar.setMin(0);
            seekBar.setMax(1000);
            int value = (int) ((cv * 1000) / range);
            seekBar.setProgress(value);
        }
    }
}
