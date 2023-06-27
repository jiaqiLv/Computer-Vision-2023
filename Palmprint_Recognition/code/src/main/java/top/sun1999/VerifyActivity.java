package top.sun1999;

import androidx.annotation.NonNull;
import androidx.annotation.Nullable;
import androidx.appcompat.app.AlertDialog;
import androidx.appcompat.app.AppCompatActivity;
import androidx.camera.core.CameraX;
import androidx.camera.core.ImageAnalysis;
import androidx.camera.core.ImageAnalysisConfig;
import androidx.camera.core.ImageProxy;
import androidx.camera.core.Preview;
import androidx.camera.core.PreviewConfig;
import androidx.camera.core.UseCase;
import androidx.core.app.ActivityCompat;

import android.Manifest;
import android.content.Context;
import android.content.DialogInterface;
import android.content.Intent;
import android.content.pm.PackageManager;
import android.database.Cursor;
import android.graphics.Bitmap;
import android.graphics.BitmapFactory;
import android.graphics.Canvas;
import android.graphics.Color;
import android.graphics.ImageFormat;
import android.graphics.Matrix;
import android.graphics.Paint;
import android.graphics.Rect;
import android.graphics.Typeface;
import android.graphics.YuvImage;
import android.media.ExifInterface;
import android.net.Uri;
import android.os.Bundle;
import android.os.Environment;
import android.os.Handler;
import android.os.Looper;
import android.os.Message;
import android.provider.MediaStore;

import android.util.Log;
import android.util.Size;
import android.widget.Button;
import android.widget.ImageView;
import android.widget.TextView;
import android.widget.Toast;

import androidx.core.content.ContextCompat;
import androidx.core.content.FileProvider;

import java.io.ByteArrayInputStream;
import java.io.ByteArrayOutputStream;
import java.io.File;
import java.io.FileInputStream;
import java.io.FileOutputStream;
import java.io.IOException;
import java.io.InputStream;
import java.io.OutputStream;
import java.lang.ref.SoftReference;
import java.nio.ByteBuffer;
import java.text.SimpleDateFormat;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.Date;
import java.util.List;
import java.util.Locale;
import java.util.concurrent.atomic.AtomicBoolean;

import static java.lang.Math.PI;
import static java.lang.Math.max;
import static java.lang.Math.pow;

import org.pytorch.IValue;
import org.pytorch.MemoryFormat;
import org.pytorch.Module;
import org.pytorch.Tensor;
import org.pytorch.torchvision.TensorImageUtils;

public class VerifyActivity extends AppCompatActivity {
    private static final String[] PERMISSIONS = {
            Manifest.permission.READ_EXTERNAL_STORAGE,
            Manifest.permission.CAMERA,
            Manifest.permission.WRITE_EXTERNAL_STORAGE
    };
    private ImageView resultImageView;

    private TextView thresholdTextview;
    private TextView tvInfo;
    private final double threshold = 0.35;
    private final double nms_threshold = 0.7;


    private long startTime = 0;
    private long endTime = 0;
    private int width;
    private int height;
    private static final Paint boxPaint = new Paint();

    private Module module;

    @Override
    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        ActivityCompat.requestPermissions(this, PERMISSIONS, PackageManager.PERMISSION_GRANTED);
        while ((ContextCompat.checkSelfPermission(this.getApplicationContext(), PERMISSIONS[0]) == PackageManager.PERMISSION_DENIED
                || ContextCompat.checkSelfPermission(this.getApplicationContext(), PERMISSIONS[1]) == PackageManager.PERMISSION_DENIED)) {
            try {
                wait(1000);
            } catch (InterruptedException e) {
                e.printStackTrace();
            }
        }
        setContentView(R.layout.activity_main);
        YOLOv4.init(getAssets());
        resultImageView = findViewById(R.id.imageView);
        thresholdTextview = findViewById(R.id.valTxtView);
        tvInfo = findViewById(R.id.tv_info);

        thresholdTextview.setText("在掌纹库中进行匹配相似的掌纹");

        Button inference = findViewById(R.id.button);
        inference.setOnClickListener(view -> {
            finish();
        });

        startCamera();
        try {
            module = Module.load(assetFilePath(this, "best.pt"));
        } catch (IOException e) {
            e.printStackTrace();
        }
    }


    /**
     * Copies specified asset to the file in /files app directory and returns this file absolute path.
     *
     * @return absolute file path
     */
    public static String assetFilePath(Context context, String assetName) throws IOException {
        File file = new File(context.getFilesDir(), assetName);
        if (file.exists() && file.length() > 0) {
            return file.getAbsolutePath();
        }

        try (InputStream is = context.getAssets().open(assetName)) {
            try (OutputStream os = new FileOutputStream(file)) {
                byte[] buffer = new byte[4 * 1024];
                int read;
                while ((read = is.read(buffer)) != -1) {
                    os.write(buffer, 0, read);
                }
                os.flush();
            }
            return file.getAbsolutePath();
        }
    }


    private void startCamera() {
        CameraX.unbindAll();

        PreviewConfig previewConfig = new PreviewConfig.Builder()
                .setLensFacing(CameraX.LensFacing.BACK)
                .setTargetResolution(new Size(416, 416))  // 分辨率
                .build();

        Preview preview = new Preview(previewConfig);
        DetectAnalyzer detectAnalyzer = new DetectAnalyzer();
        CameraX.bindToLifecycle(this, preview, gainAnalyzer(detectAnalyzer));

    }


    private UseCase gainAnalyzer(DetectAnalyzer detectAnalyzer) {
        ImageAnalysisConfig.Builder analysisConfigBuilder = new ImageAnalysisConfig.Builder();
        analysisConfigBuilder.setImageReaderMode(ImageAnalysis.ImageReaderMode.ACQUIRE_LATEST_IMAGE);
        analysisConfigBuilder.setTargetResolution(new Size(416, 416));  // 输出预览图像尺寸
        ImageAnalysisConfig config = analysisConfigBuilder.build();
        ImageAnalysis analysis = new ImageAnalysis(config);
        analysis.setAnalyzer(detectAnalyzer);
        return analysis;
    }


    private class DetectAnalyzer implements ImageAnalysis.Analyzer {
        @Override
        public void analyze(ImageProxy image, final int rotationDegrees) {
//            if (detecting.get() || detectPhoto.get()) {
//                return;
//            }
//            detecting.set(true);
            final Bitmap bitmapsrc = imageToBitmap(image);  // 格式转换
            Thread detectThread = new Thread(() -> {
                Matrix matrix = new Matrix();
                matrix.postRotate(rotationDegrees);
                width = bitmapsrc.getWidth();
                height = bitmapsrc.getHeight();
                Bitmap bitmap = Bitmap.createBitmap(bitmapsrc, 0, 0, width, height, matrix, false);

                startTime = System.currentTimeMillis();
                Box[] result = YOLOv4.detect(bitmap, threshold, nms_threshold);
                endTime = System.currentTimeMillis();

                final Bitmap mutableBitmap = bitmap.copy(Bitmap.Config.ARGB_8888, true);
                float strokeWidth = 4 * (float) mutableBitmap.getWidth() / 800;
                float textSize = 30 * (float) mutableBitmap.getWidth() / 800;

                Canvas canvas = new Canvas(mutableBitmap);
                boxPaint.setAlpha(255);
                boxPaint.setTypeface(Typeface.SANS_SERIF);
                boxPaint.setStyle(Paint.Style.STROKE);
                boxPaint.setStrokeWidth(strokeWidth);
                boxPaint.setTextSize(textSize);
                for (Box box : result) {
                    boxPaint.setColor(box.getColor());
                    boxPaint.setStyle(Paint.Style.FILL);
                    String score = Integer.toString((int) (box.getScore() * 100));
                    canvas.drawText(box.getLabel() + " [" + score + "%]",
                            box.x0 - strokeWidth, box.y0 - strokeWidth
                            , boxPaint);
                    boxPaint.setStyle(Paint.Style.STROKE);
                    canvas.drawRect(box.getRect(), boxPaint);
                }

                String res;
                Bitmap roi = Util.extractROI(mutableBitmap, result, true);

                if (Util.names.size() == 0) {
                    res = "请录入掌纹";
                } else {
                    if (roi == null) {
                        res = "请将摄像头对准手掌";
                    } else {
                        final Tensor inputTensor = TensorImageUtils.bitmapToFloat32Tensor(roi,
                                TensorImageUtils.TORCHVISION_NORM_MEAN_RGB, TensorImageUtils.TORCHVISION_NORM_STD_RGB, MemoryFormat.CHANNELS_LAST);

                        // running the model
                        final Tensor outputTensor = module.forward(IValue.from(inputTensor)).toTensor();

                        float[] vec = outputTensor.getDataAsFloatArray();
                        float sum = 0;
                        for (float tmp : vec) {
                            sum += tmp * tmp;
                        }
                        sum = (float) Math.sqrt(sum);

                        List<Double> dot = new ArrayList<>();
                        double tmpDot = 0f;
                        for (double[] vect : Util.vecs) {
                            for (int i = 0; i < vect.length; i++) {
                                tmpDot += vec[i] / sum * vect[i];
                            }
                            dot.add(tmpDot);
                            tmpDot = 0f;
                        }
                        double maxDot = -1;
                        for (int i = 0; i < dot.size(); i++) {
                            if (dot.get(i) > maxDot) {
                                maxDot = dot.get(i);
                            }
                        }
                        String name = Util.names.get(dot.indexOf(maxDot));

                        endTime = System.currentTimeMillis();
//                        String res1 = dot > 0.432 ? "掌纹匹配成功" : "掌纹匹配失败";
                        res = String.format(Locale.CHINESE,
                                "%s，你被我认出来了！\n相似度: %.3f\nImgSize: %dx%d\nUseTime: %d ms",
                                name, maxDot, mutableBitmap.getHeight(), mutableBitmap.getWidth(), endTime - startTime);

                    }

                }
                runOnUiThread(() -> {
                    resultImageView.setImageBitmap(mutableBitmap);
                    tvInfo.setText(res);
                });
            }, "detect");
            detectThread.start();
        }

        private Bitmap imageToBitmap(ImageProxy image) {
            ImageProxy.PlaneProxy[] planes = image.getPlanes();
            ImageProxy.PlaneProxy y = planes[0];
            ImageProxy.PlaneProxy u = planes[1];
            ImageProxy.PlaneProxy v = planes[2];
            ByteBuffer yBuffer = y.getBuffer();
            ByteBuffer uBuffer = u.getBuffer();
            ByteBuffer vBuffer = v.getBuffer();
            int ySize = yBuffer.remaining();
            int uSize = uBuffer.remaining();
            int vSize = vBuffer.remaining();
            byte[] nv21 = new byte[ySize + uSize + vSize];
            // U and V are swapped
            yBuffer.get(nv21, 0, ySize);
            vBuffer.get(nv21, ySize, vSize);
            uBuffer.get(nv21, ySize + vSize, uSize);

            YuvImage yuvImage = new YuvImage(nv21, ImageFormat.NV21, image.getWidth(), image.getHeight(), null);
            ByteArrayOutputStream out = new ByteArrayOutputStream();
            yuvImage.compressToJpeg(new Rect(0, 0, yuvImage.getWidth(), yuvImage.getHeight()), 100, out);
            byte[] imageBytes = out.toByteArray();

            return BitmapFactory.decodeByteArray(imageBytes, 0, imageBytes.length);
        }

    }


    @Override
    protected void onDestroy() {
        CameraX.unbindAll();
        super.onDestroy();
    }


    @Override
    public void onRequestPermissionsResult(int requestCode, @NonNull String[] permissions, @NonNull int[] grantResults) {
        super.onRequestPermissionsResult(requestCode, permissions, grantResults);
        for (int result : grantResults) {
            if (result != PackageManager.PERMISSION_GRANTED) {
                this.finish();
            }
        }
    }

//    @Override
//    protected void onActivityResult(int requestCode, int resultCode, @Nullable Intent data) {
//        super.onActivityResult(requestCode, resultCode, data);
////        if (data == null) {
////            return;
////        }
//
//        detectPhoto.set(true);
//
////        byte[] photoByUrl = getPhotoByUrl(currentPhotoPath);
////        Bitmap image = byteToBitmap(getPhotoByUrl(currentPhotoPath));
////        Bitmap image = (Bitmap) photoByUrl;
//        Bitmap image = getPicture(data.getData());
//        //根据Uri显示本地图片
////        photoImageView.setImageURI(uri);
////        Log.d("success","路径：" + uri.toString());
//
//
////        Bundle bundle = data.getExtras(); // 从data中取出传递回来缩略图的信息，图片质量差，适合传递小图片
////        Bitmap image = (Bitmap) bundle.get("data"); // 将data中的信息流解析为Bitmap类型
//
////        Bitmap image = getPicture(data.getData());
//
//        startTime = System.currentTimeMillis();
//        Box[] result = YOLOv4.detect(image, threshold, nms_threshold);
//        Bitmap tmpimage = Util.extractROI(image, result);
//        resultImageView.setImageBitmap(tmpimage);
//        if (tmpimage == null) {
//            tvInfo.setText("Yolo 识别失败");
////            detectPhoto.set(false);
//            return;
//        }
//
//
//        // preparing input tensor
//        final Tensor inputTensor = TensorImageUtils.bitmapToFloat32Tensor(tmpimage,
//                TensorImageUtils.TORCHVISION_NORM_MEAN_RGB, TensorImageUtils.TORCHVISION_NORM_STD_RGB, MemoryFormat.CHANNELS_LAST);
//
//        // running the model
//        final Tensor outputTensor = module.forward(IValue.from(inputTensor)).toTensor();
//
//
//        if (true) {
////            if (myVec == null) {
//            // getting tensor content as java array of floats
//            myVec = outputTensor.getDataAsFloatArray();
//
//            float sum = 0;
//            for (float tmp : myVec) {
//                sum += tmp * tmp;
//            }
//            sum = (float) Math.sqrt(sum);
//            for (int i = 0; i < myVec.length; i++) {
//                myVec[i] /= sum;
//            }
//            endTime = System.currentTimeMillis();
//            Log.e("myVec", myVec.toString());
//            tvInfo.setText(String.format(Locale.CHINESE,
//                    "ImgSize: %dx%d\nUseTime: %d ms\n特征向量已保存",
//                    image.getHeight(), image.getWidth(), endTime - startTime));
//        } else {
//            float[] vec = outputTensor.getDataAsFloatArray();
//            float sum = 0;
//            for (float tmp : vec) {
//                sum += tmp * tmp;
//            }
//            sum = (float) Math.sqrt(sum);
//
//            float dot = 0;
//            for (int i = 0; i < myVec.length; i++) {
//                dot += vec[i] / sum * myVec[i];
//            }
//            endTime = System.currentTimeMillis();
//            String res = dot > 0.432 ? "掌纹匹配成功" : "掌纹匹配失败";
//            tvInfo.setText(String.format(Locale.CHINESE,
//                    "ImgSize: %dx%d\nUseTime: %d ms\n相似度: %.3f %s ",
//                    image.getHeight(), image.getWidth(), endTime - startTime, dot, res));
//
//        }
////        detectPhoto.set(false);
//    }

    public void makeDialog(String text) {
        //创建一个警告对话框
        AlertDialog.Builder builder = new AlertDialog.Builder(this);

        builder.setTitle("提示");
        builder.setMessage(text);
        builder.setPositiveButton("确定", new DialogInterface.OnClickListener() {
            @Override
            public void onClick(DialogInterface dialog, int which) {

            }
        });

        AlertDialog alertDialog = builder.create();//这个方法可以返回一个alertDialog对象
        alertDialog.show();

    }


    public Bitmap getPicture(Uri selectedImage) {
        String[] filePathColumn = {MediaStore.Images.Media.DATA};
        Cursor cursor = this.getContentResolver().query(selectedImage, filePathColumn, null, null, null);
        cursor.moveToFirst();
        int columnIndex = cursor.getColumnIndex(filePathColumn[0]);
        String picturePath = cursor.getString(columnIndex);
        cursor.close();
        Bitmap bitmap = BitmapFactory.decodeFile(picturePath);
        int rotate = readPictureDegree(picturePath);
        return rotateBitmapByDegree(bitmap, rotate);
    }

    public int readPictureDegree(String path) {
        int degree = 0;
        try {
            ExifInterface exifInterface = new ExifInterface(path);
            int orientation = exifInterface.getAttributeInt(ExifInterface.TAG_ORIENTATION, ExifInterface.ORIENTATION_NORMAL);
            switch (orientation) {
                case ExifInterface.ORIENTATION_ROTATE_90:
                    degree = 90;
                    break;
                case ExifInterface.ORIENTATION_ROTATE_180:
                    degree = 180;
                    break;
                case ExifInterface.ORIENTATION_ROTATE_270:
                    degree = 270;
                    break;
            }
        } catch (IOException e) {
            e.printStackTrace();
        }
        return degree;
    }

    public Bitmap rotateBitmapByDegree(Bitmap bm, int degree) {
        Matrix matrix = new Matrix();
        matrix.postRotate(degree);

        Bitmap returnBm = Bitmap.createBitmap(bm, 0, 0, bm.getWidth(), bm.getHeight(), matrix, true);

        if (returnBm == null) {
            returnBm = bm;
        }
        if (bm != returnBm) {
            bm.recycle();
        }
        return returnBm;
    }

    String currentPhotoPath;

    private File createImageFile() throws IOException {
        // Create an image file name
        String timeStamp = new SimpleDateFormat("yyyyMMdd_HHmmss").format(new Date());
        String imageFileName = "JPEG_" + timeStamp + "_";
        File storageDir = getExternalFilesDir(Environment.DIRECTORY_PICTURES);
        File image = File.createTempFile(
                imageFileName,  /* prefix */
                ".jpg",         /* suffix */
                storageDir      /* directory */
        );

        // Save a file: path for use with ACTION_VIEW intents
        currentPhotoPath = image.getAbsolutePath();
        return image;
    }

    public static byte[] getPhotoByUrl(String photoUrl) {
        File file = new File(photoUrl);
        if (!file.exists()) {
            return null;
        }
        try {
            FileInputStream fis = new FileInputStream(file);
            ByteArrayOutputStream out = new ByteArrayOutputStream();
            byte[] temp = new byte[1024];
            int size = 0;
            while ((size = fis.read(temp)) != -1) {
                out.write(temp, 0, size);
            }
            fis.close();
            return out.toByteArray();
        } catch (Exception e) {
            e.printStackTrace();
        }

        return null;
    }

    public static Bitmap byteToBitmap(byte[] imgByte) {
        InputStream input = null;
        Bitmap bitmap = null;
        BitmapFactory.Options options = new BitmapFactory.Options();
        options.inSampleSize = 1;
        input = new ByteArrayInputStream(imgByte);
        SoftReference softRef = new SoftReference(BitmapFactory.decodeStream(
                input, null, options));  //软引用防止OOM
        bitmap = (Bitmap) softRef.get();
        if (imgByte != null) {
            imgByte = null;
        }

        try {
            if (input != null) {
                input.close();
            }
        } catch (IOException e) {
            // 异常捕获
            e.printStackTrace();
        }
        return bitmap;
    }
}
