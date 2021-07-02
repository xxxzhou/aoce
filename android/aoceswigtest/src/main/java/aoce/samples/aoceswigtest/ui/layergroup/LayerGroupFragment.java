package aoce.samples.aoceswigtest.ui.layergroup;

import android.os.Bundle;
import android.view.LayoutInflater;
import android.view.View;
import android.view.ViewGroup;

import com.google.android.material.tabs.TabLayout;
import com.google.android.material.tabs.TabLayoutMediator;

import androidx.annotation.NonNull;
import androidx.annotation.Nullable;
import androidx.fragment.app.Fragment;
import androidx.viewpager2.widget.ViewPager2;
import aoce.samples.aoceswigtest.DataManager;
import aoce.samples.aoceswigtest.R;
import butterknife.BindView;
import butterknife.ButterKnife;

public class LayerGroupFragment extends Fragment {
    @BindView(R.id.viewpage)
    ViewPager2 viewPager2;
    @BindView(R.id.homeTabLayout)
    TabLayout tabLayout;

    public View onCreateView(@NonNull LayoutInflater inflater,
                             ViewGroup container, Bundle savedInstanceState) {
        View view = inflater.inflate(R.layout.fragment_layer_group, container, false);
        ButterKnife.bind(this, view);
        return view;
    }

    @Override
    public void onViewCreated(View view, @Nullable Bundle savedInstanceState) {
        super.onViewCreated(view, savedInstanceState);

        LayerGroupAdapter adapter = new LayerGroupAdapter(this);
        viewPager2.setAdapter(adapter);
        viewPager2.registerOnPageChangeCallback(new ViewPager2.OnPageChangeCallback() {
            @Override
            public void onPageScrolled(int position, float positionOffset, int positionOffsetPixels) {
                super.onPageScrolled(position, positionOffset, positionOffsetPixels);
            }
        });
        new TabLayoutMediator(tabLayout, viewPager2, new TabLayoutMediator.TabConfigurationStrategy() {
            @Override
            public void onConfigureTab(@NonNull TabLayout.Tab tab, int position) {
                //设置TabLayout的显示
                //tab:当前处于选中状态的Tab对象 position:当前Tab所处的位置
                tab.setText(DataManager.getInstance().getIndex(position).title);
            }
        }).attach();
    }
}