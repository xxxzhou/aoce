package aoce.samples.aoceswigtest;

import java.util.ArrayList;

import androidx.lifecycle.MutableLiveData;
import androidx.lifecycle.ViewModel;

public class TestViewModel extends ViewModel {

    private MutableLiveData<Integer> id = null;
    private ArrayList<String> names = new ArrayList<String>();

    public String getName(int index){
        if(index<0 || index>names.size()){
            return null;
        }
        return names.get(index);
    }

    public MutableLiveData<Integer> getId(){
        return id;
    }
}
