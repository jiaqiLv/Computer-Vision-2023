package top.sun1999;

import android.content.Context;
import android.graphics.Bitmap;
import android.graphics.Canvas;
import android.graphics.Color;
import android.graphics.Matrix;
import android.graphics.Paint;
import android.graphics.Typeface;
import android.util.Log;

import java.io.BufferedWriter;
import java.io.FileNotFoundException;
import java.io.FileOutputStream;
import java.io.IOException;
import java.io.OutputStreamWriter;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.HashMap;
import java.util.List;
import java.util.Map;

public class Util {

    public static List<String> names = new ArrayList<>();
    public static List<double[]> vecs = new ArrayList<>();

    public static List<Double> onePoint(double x, double y, Double angle) {
        Double X = x * Math.cos(angle) + y * Math.sin(angle);
        Double Y = y * Math.cos(angle) + x * Math.sin(angle);
        List<Double> point = new ArrayList<>();
        point.add(X);
        point.add(Y);
        return point;
    }

    public static Bitmap extractROI(Bitmap img, Box[] yoloRes) {
        Integer H = img.getHeight(), W = img.getWidth();
        float x1 = -1;
        float y1 = 0;
        float x2 = 0;
        float y2 = 0;
        float x3 = 0;
        float y3 = 0;

        if (yoloRes.length != 3) {
            return null;
        }
        for (Box box : yoloRes) {
            if (box.getLabel().equals("double gap")) {
                if (x1 == -1) {
                    x1 = (box.x0 + box.x1) / 2;
                    y1 = (box.y1 + box.y0) / 2;
                } else {
                    x2 = (box.x0 + box.x1) / 2;
                    y2 = (box.y1 + box.y0) / 2;
                }
            } else {
                x3 = (box.x0 + box.x1) / 2;
                y3 = (box.y1 + box.y0) / 2;
            }
        }
        if (x2 == 0 || x3 == 0) {
            return null;
        }

        float x0 = (x1 + x2) / 2;
        float y0 = (y1 + y2) / 2;

        double unitLen = Math.sqrt(Math.pow(x2 - x1, 2) + Math.pow(y2 - y1, 2));

        float k1 = (y1 - y2) / (x1 - x2); // line AB
        float b1 = y1 - k1 * x1;

        float k2 = (-1) / k1;
        float b2 = y3 - k2 * x3;

        float tmpX = (b2 - b1) / (k1 - k2);  // AB中点
        float tmpY = k1 * tmpX + b1;

        float vec0 = x3 - tmpX;
        float vec1 = y3 - tmpY;
        double sidLen = Math.sqrt(vec0 * vec0 + vec1 * vec1); // C到AB中点距离
        vec0 = (float) (vec0 / sidLen);     //OC单位向量
        vec1 = (float) (vec1 / sidLen);

        double angle;
        if (vec1 < 0 && vec0 > 0) angle = Math.PI / 2 - Math.acos(vec0);
        else if (vec1 < 0 && vec0 < 0) angle = Math.acos(-vec0) - Math.PI / 2;
        else if (vec1 >= 0 && vec0 > 0) angle = Math.acos(vec0) - Math.PI / 2;
        else angle = Math.PI / 2 - Math.acos(-vec0);

        Matrix matrix = new Matrix();
        matrix.postRotate((float) (-angle / Math.PI * 180));
        img = Bitmap.createBitmap(img, 0, 0, img.getWidth(), img.getHeight(), matrix, true);

        List<Double> xy0 = onePoint(x0 - W / 2, y0 - H / 2, angle);
        x0 = (float) (xy0.get(0) + img.getWidth() / 2);
        y0 = (float) (xy0.get(1) + img.getHeight() / 2);

        Matrix matrix1 = new Matrix();
        matrix1.postScale((float) (224 / (unitLen * 5 / 2)), (float) (224 / (unitLen * 5 / 2)));
        img = Bitmap.createBitmap(img,
                (int) Math.round(x0 - unitLen * 5 / 4),
                (int) Math.round(y0 + unitLen / 4),
                (int) Math.round(unitLen * 5 / 2),
                (int) Math.round(unitLen * 5 / 2),
                matrix1, true);
        Log.e("size", String.valueOf(img.getHeight()));
        Log.e("width", String.valueOf(img.getWidth()));
        return img;
    }

    public static Bitmap extractROI(Bitmap img, Box[] yoloRes, Boolean draw) {
        Integer H = img.getHeight(), W = img.getWidth();
        float x1 = -1;
        float y1 = 0;
        float x2 = 0;
        float y2 = 0;
        float x3 = 0;
        float y3 = 0;

        if (yoloRes.length != 3) {
            return null;
        }
        for (Box box : yoloRes) {
            if (box.getLabel().equals("double gap")) {
                if (x1 == -1) {
                    x1 = (box.x0 + box.x1) / 2;
                    y1 = (box.y1 + box.y0) / 2;
                } else {
                    x2 = (box.x0 + box.x1) / 2;
                    y2 = (box.y1 + box.y0) / 2;
                }
            } else {
                x3 = (box.x0 + box.x1) / 2;
                y3 = (box.y1 + box.y0) / 2;
            }
        }
        if (x2 == 0 || x3 == 0) {
            return null;
        }

        float x0 = (x1 + x2) / 2;
        float y0 = (y1 + y2) / 2;


        double unitLen = Math.sqrt(Math.pow(x2 - x1, 2) + Math.pow(y2 - y1, 2));

        float k1 = (y1 - y2) / (x1 - x2); // line AB
        float b1 = y1 - k1 * x1;

        float k2 = (-1) / k1;
        float b2 = y3 - k2 * x3;

        float tmpX = (b2 - b1) / (k1 - k2);  // AB中点
        float tmpY = k1 * tmpX + b1;

        float vec0 = x3 - tmpX;
        float vec1 = y3 - tmpY;
        double sidLen = Math.sqrt(vec0 * vec0 + vec1 * vec1); // C到AB中点距离
        vec0 = (float) (vec0 / sidLen);     //OC单位向量
        vec1 = (float) (vec1 / sidLen);

        double angle;
        if (vec1 < 0 && vec0 > 0) angle = Math.PI / 2 - Math.acos(vec0);
        else if (vec1 < 0 && vec0 < 0) angle = Math.acos(-vec0) - Math.PI / 2;
        else if (vec1 >= 0 && vec0 > 0) angle = Math.acos(vec0) - Math.PI / 2;
        else angle = Math.PI / 2 - Math.acos(-vec0);

        Matrix matrix = new Matrix();
        matrix.postRotate((float) (-angle / Math.PI * 180));
        Bitmap newimg = Bitmap.createBitmap(img, 0, 0, img.getWidth(), img.getHeight(), matrix, true);


        Canvas canvas = new Canvas(img);
        Paint boxPaint = new Paint();
        float strokeWidth = 4 * (float) img.getWidth() / 800;
        float textSize = 30 * (float) img.getWidth() / 800;
        boxPaint.setColor(Color.argb(255, 255, 165, 0));
        boxPaint.setAlpha(255);
        boxPaint.setTypeface(Typeface.SANS_SERIF);
        boxPaint.setStyle(Paint.Style.STROKE);
        boxPaint.setStrokeWidth(strokeWidth);
        boxPaint.setTextSize(textSize);
        canvas.drawLine(x1, y1, x2, y2, boxPaint);
        canvas.drawLine(x0, y0, x3, y3, boxPaint);

        List<Double> xy0 = onePoint(x0 - W / 2, y0 - H / 2, angle);
        x0 = (float) (xy0.get(0) + newimg.getWidth() / 2);
        y0 = (float) (xy0.get(1) + newimg.getHeight() / 2);

        Matrix matrix1 = new Matrix();
        matrix1.postScale((float) (224 / (unitLen * 5 / 2)), (float) (224 / (unitLen * 5 / 2)));
        try {
            newimg = Bitmap.createBitmap(newimg,
                    (int) Math.round(x0 - unitLen * 5 / 4),
                    (int) Math.round(y0 + unitLen / 4),
                    (int) Math.round(unitLen * 5 / 2),
                    (int) Math.round(unitLen * 5 / 2),
                    matrix1, true);
        } catch (Exception e) {
            return null;
        }

        return newimg;
    }


    public static void main(String[] args) {

    }

}
