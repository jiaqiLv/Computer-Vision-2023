package top.sun1999;

import androidx.appcompat.app.AlertDialog;
import androidx.appcompat.app.AppCompatActivity;

import android.content.DialogInterface;
import android.content.Intent;
import android.os.Bundle;
import android.view.View;
import android.widget.Button;

import java.io.BufferedReader;
import java.io.BufferedWriter;
import java.io.FileInputStream;
import java.io.FileNotFoundException;
import java.io.FileOutputStream;
import java.io.IOException;
import java.io.InputStreamReader;
import java.io.OutputStreamWriter;
import java.util.Arrays;

public class NewMainActivity extends AppCompatActivity {

    Button beginVerifyButton;
    Button addPalmButton;
    Button multiRecognizeButton;
    Button deletePalmprint;

    @Override
    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        setContentView(R.layout.activity_new_main);

        beginVerifyButton = findViewById(R.id.beginVerifying);
        addPalmButton = findViewById(R.id.addPalmButton);
        multiRecognizeButton = findViewById(R.id.multiRecognize);
        deletePalmprint = findViewById(R.id.deletePalmprint);

        multiRecognizeButton.setOnClickListener(new View.OnClickListener() {
            @Override
            public void onClick(View v) {
                Intent i = new Intent(NewMainActivity.this, VerifyActivity.class);
                startActivity(i);
            }
        });

        addPalmButton.setOnClickListener(new View.OnClickListener() {
            @Override
            public void onClick(View v) {
                Intent i = new Intent(NewMainActivity.this, AddPalmActicity.class);
                startActivity(i);
            }
        });

        beginVerifyButton.setOnClickListener(new View.OnClickListener() {
            @Override
            public void onClick(View v) {
                Intent i = new Intent(NewMainActivity.this, SingleVerifyActivity.class);
                startActivity(i);
            }
        });

        deletePalmprint.setOnClickListener(new View.OnClickListener() {
            @Override
            public void onClick(View v) {
                deletePalmprint();
            }
        });


        readVecs();
    }

    public void readVecs() {
        FileInputStream input = null;
        BufferedReader bufferedReader = null;
        try {
            input = openFileInput("vecs.txt");
            bufferedReader = new BufferedReader(new InputStreamReader(input));
            String s = bufferedReader.readLine();
            while (s != null) {
                if (s.indexOf('[') >= 0) {
                    s = s.substring(1, s.length() - 1);
                    String[] split = s.split(",");
                    double[] doubles = Arrays.stream(split).mapToDouble(Double::parseDouble).toArray();
                    Util.vecs.add(doubles);
                } else {
                    Util.names.add(s);
                }
                s = bufferedReader.readLine();
            }
        } catch (FileNotFoundException e) {
            e.printStackTrace();
        } catch (IOException e) {
            e.printStackTrace();
        } finally {
            try {
                bufferedReader.close();
                input.close();
            } catch (IOException e) {
                e.printStackTrace();
            }
        }

    }

    public void deletePalmprint() {
        String[] names = Util.names.toArray(new String[Util.names.size()]);
        boolean checkedItems[] = new boolean[names.length];
        for (int i = 0; i < names.length; i++) {
            checkedItems[i] = false;
        }
        AlertDialog dialog = new AlertDialog.Builder(this)
                .setTitle("选择您要删除的掌纹")//设置对话框的标题
                .setMultiChoiceItems(names, checkedItems, new DialogInterface.OnMultiChoiceClickListener() {
                    @Override
                    public void onClick(DialogInterface dialog, int which, boolean isChecked) {
                        checkedItems[which] = isChecked;
                    }
                })
                .setNegativeButton("取消", new DialogInterface.OnClickListener() {
                    @Override
                    public void onClick(DialogInterface dialog, int which) {
                        dialog.dismiss();
                    }
                })
                .setPositiveButton("确定", new DialogInterface.OnClickListener() {
                    @Override
                    public void onClick(DialogInterface dialog, int which) {
                        for (int i = checkedItems.length - 1; i >= 0; i--) {
                            if (checkedItems[i]) {
                                Util.names.remove(i);
                                Util.vecs.remove(i);


                                FileOutputStream outputStream = null;
                                BufferedWriter bufferedWriter = null;
                                try {
                                    outputStream = openFileOutput("vecs.txt", MODE_PRIVATE);
                                    bufferedWriter = new BufferedWriter(new
                                            OutputStreamWriter(outputStream));

                                    for (double[] arr : Util.vecs) {
                                        String strArr[] = Arrays.stream(arr).mapToObj(String::valueOf).toArray(String[]::new);
                                        bufferedWriter.write(Arrays.toString(strArr));
                                        bufferedWriter.write('\n');
                                    }
                                    for (String s : Util.names) {
                                        bufferedWriter.write(s);
                                        bufferedWriter.write('\n');
                                    }
                                } catch (FileNotFoundException e) {
                                    e.printStackTrace();
                                } catch (IOException e) {
                                    e.printStackTrace();
                                } finally {
                                    try {
                                        assert bufferedWriter != null;
                                        bufferedWriter.close();
                                        outputStream.close();
                                    } catch (IOException e) {
                                        e.printStackTrace();
                                    }
                                }
                            }
                        }
                        dialog.dismiss();
                    }
                }).create();
        dialog.show();

    }

}