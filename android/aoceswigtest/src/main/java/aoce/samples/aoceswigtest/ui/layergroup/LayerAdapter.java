package aoce.samples.aoceswigtest.ui.layergroup;

import android.content.Context;
import android.view.View;
import android.view.ViewGroup;
import android.widget.TextView;

import androidx.annotation.NonNull;
import androidx.recyclerview.widget.RecyclerView;
import aoce.samples.aoceswigtest.DataManager;
import aoce.samples.aoceswigtest.R;
import butterknife.BindView;
import butterknife.ButterKnife;

public class LayerFragmentAdapter extends RecyclerView.Adapter<RecyclerView.ViewHolder> {
    Context context = null;
    DataManager.LayerGroup layerGroup = null;
    public LayerFragmentAdapter(Context context, int groupIndex) {
        this.context = context;
        this.layerGroup = DataManager.getInstance().getIndex(groupIndex);;
    }

    @NonNull
    @Override
    public RecyclerView.ViewHolder onCreateViewHolder(@NonNull ViewGroup parent, int viewType) {
        View view = View.inflate(parent.getContext(), R.layout.item_layer,null);
        return new LayerViewHolder(view);
    }

    @Override
    public void onBindViewHolder(@NonNull RecyclerView.ViewHolder holder, int position) {
        LayerViewHolder item = (LayerViewHolder)holder;
        item.bindItem(layerGroup.layers.get(position));
    }

    @Override
    public int getItemCount() {
        return layerGroup.layers.size();
    }

    class LayerViewHolder extends RecyclerView.ViewHolder{
        @BindView(R.id.layerName)
        TextView textView;
        LayerViewHolder(View view){
            super(view);
            ButterKnife.bind(this,view);
        }

        private void bindItem(DataManager.LayerItem layerItem){
            textView.setText(layerItem.name);
        }
    }
}
