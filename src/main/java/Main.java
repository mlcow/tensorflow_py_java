import org.tensorflow.SavedModelBundle;
import org.tensorflow.Session;
import org.tensorflow.Tensor;

import java.nio.DoubleBuffer;
import java.util.Arrays;
import java.util.List;

//
// https://stackoverflow.com/questions/46098863/how-to-import-an-saved-tensorflow-model-train-using-tf-estimator-and-predict-on

public class Main {
    public static void main(String[] args) {
        if(args.length < 1) {
            System.out.println("Pass argument model_dir: example: export2/1520404045");
        }
        Session session = SavedModelBundle.load(args[0], "serve").session();

        Tensor x = Tensor.create(new long[]{2,1}, DoubleBuffer.wrap(new double[] {
                1.0d, 2.0d
        }));

        Tensor y = Tensor.create(new long[]{2,1}, DoubleBuffer.wrap(new double[] {
                1.0d, 2.0d
        }));

        List<Tensor<?>> result = session.runner()
                .feed("x", x)
                .feed("y", y)
                .fetch("add:0")
                .run();

        System.out.println(result.size());
        result.forEach(t -> {
            System.out.println(t);
            System.out.println(t.dataType());
            double[][] ans = new double[2][1];
            t.copyTo(ans);
            System.out.println(Arrays.deepToString(ans));
        });
    }
}
