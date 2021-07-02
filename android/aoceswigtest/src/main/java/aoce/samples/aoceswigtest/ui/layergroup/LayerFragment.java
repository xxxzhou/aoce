package aoce.samples.aoceswigtest.ui.layergroup;

import android.os.Bundle;
import android.view.LayoutInflater;
import android.view.View;
import android.view.ViewGroup;

import androidx.fragment.app.Fragment;
import androidx.annotation.NonNull;
import androidx.annotation.Nullable;
import androidx.recyclerview.widget.GridLayoutManager;
import androidx.recyclerview.widget.RecyclerView;
import aoce.samples.aoceswigtest.DataManager;
import aoce.samples.aoceswigtest.R;
import butterknife.BindView;
import butterknife.ButterKnife;

public class LayerGroupFragment extends Fragment {
    @BindView(R.id.recyclerView)
    RecyclerView recyclerview;
    public int groupIndex = 0;

    @Nullable
    @Override
    public View onCreateView(@NonNull LayoutInflater inflater,
                             @Nullable ViewGroup container,
                             @Nullable Bundle savedInstanceState) {
        View view = inflater.inflate(R.layout.fragment_layer_group, container, false);
        ButterKnife.bind(this,view);
        return view;
    }

    @Override
    public void onViewCreated(View view, @Nullable Bundle savedInstanceState) {
        super.onViewCreated(view, savedInstanceState);
        Bundle args = getArguments();
        if (args != null) {
            groupIndex = args.getInt("groupIndex");
        }
        LayerFragmentAdapter adapter = new LayerFragmentAdapter(getActivity(), groupIndex);
        GridLayoutManager layoutManager = new GridLayoutManager(getActivity(),1);
        recyclerview.setLayoutManager(layoutManager);
        recyclerview.setAdapter(adapter);
    }
}