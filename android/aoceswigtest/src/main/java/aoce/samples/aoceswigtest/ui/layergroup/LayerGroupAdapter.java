package aoce.samples.aoceswigtest.ui.layergroup;

import android.os.Bundle;

import java.util.ArrayList;

import androidx.annotation.NonNull;
import androidx.fragment.app.Fragment;
import androidx.viewpager2.adapter.FragmentStateAdapter;
import aoce.samples.aoceswigtest.DataManager;

public class LayerGroupAdapter extends FragmentStateAdapter {
    int groupCount = 0;
    private ArrayList<Fragment> fragments = new ArrayList<>();
    public LayerGroupAdapter(Fragment parent){
        super(parent);
        groupCount = DataManager.getInstance().getGroupCount();
        for (int i=0;i<groupCount;i++){
            LayerFragment fragment = new LayerFragment();
            Bundle args = new Bundle();
            args.putInt("groupIndex",i);
            fragment.setArguments(args);
            fragments.add(fragment);
        }
    }

    @NonNull
    @Override
    public Fragment createFragment(int position) {
        return fragments.get(position);
    }

    @Override
    public int getItemCount() {
        return DataManager.getInstance().getGroupCount();
    }
}
