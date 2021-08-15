package aoce.android.library;

import android.content.res.AssetManager;
import android.graphics.Bitmap;
import android.graphics.BitmapFactory;

import java.io.IOException;
import java.io.InputStream;

public class AoceHelper {
    public static Class<?> getPrimitive(final Class<?> primitiveType) {
        if (Boolean.class.equals(primitiveType)) {
            return boolean.class;
        } else if (Float.class.equals(primitiveType)) {
            return float.class;
        } else if (Long.class.equals(primitiveType)) {
            return long.class;
        } else if (Integer.class.equals(primitiveType)) {
            return int.class;
        } else if (Short.class.equals(primitiveType)) {
            return short.class;
        } else if (Byte.class.equals(primitiveType)) {
            return byte.class;
        } else if (Double.class.equals(primitiveType)) {
            return double.class;
        } else if (Character.class.equals(primitiveType)) {
            return char.class;
        } else {
            return primitiveType;
        }
    }

    public static Bitmap getBitmapFromAsset(AssetManager assetManager, String filePath) {
        InputStream istr = null;
        Bitmap bitmap = null;
        try {
            istr = assetManager.open(filePath);
            bitmap = BitmapFactory.decodeStream(istr);
        } catch (IOException e) {
            // handle exception
        }
        return bitmap;
    }
}
