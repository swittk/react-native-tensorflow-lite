import { NativeModules } from 'react-native';

export type SKTFLiteModelInfoTensorInfoType = {
  name: string,
  dataType: number,
  shape: number[]
}
export type SKTFLiteModelInfoType = {
  input: SKTFLiteModelInfoTensorInfoType,
  output: SKTFLiteModelInfoTensorInfoType
}

export type SKTFLiteRunModelWithFilesArg = {
  /** The path to the tensorflow lite model to use */
  model: string,
  /** The paths to the files to evaluate*/
  files: string[],
  /** The shapes of the input tensors. CAREFUL THIS WILL CRASH YOUR APP IF NOT PROPERLY DEFINED
   * If this is not provided, the shapes are rudimentarily inferred from the model.
   */
  shapes?: number[][],

  /** 
   * If the files are to be "group"ed together (like, we input multiple files at once to the model) 
   * then it should be specified here.
  */
  groupMode?: {
    /** Number of items to `group` together in the input */
    numPerGroup?: number,
    /** Stride when enumerating the input files (defaults to 1) */
    stride?: number,
    /** 
     * Whether to concatenate via another dimension, or to simply extend the data 
     * Defaults to 'dims'.
     * In `dims` mode, image inputs are automatically resized to match the model's input tensor
     * the data is then appended end to end with each other (causing a new axis in the first dimension)
     * (Currently only `dims` is implemented)
    */
    concatMode?: 'dims' | 'extend',
    /** The axis to extend the data to in 'extend' mode (not implemented yet) */
    extendAxis?: number
  }

  /** To be decided what to do with this, maybe `image`, etc */
  fileMode?: 'image',
  /** If image scaling should be by `fill` or `fit` (defaults to fill) 
   * Fit = pad with black bars to fit the image too
  */
  imageScaleMode?: 'fill' | 'fit',
  /** if specified; [x, y, width, height] parts of the input images to crop to when processing. */
  imageCrops?: [number, number, number, number][],
  /** The mode of the crops. If 'relative' then the inputs are from 0-1, 'absolute' then the inputs are absolute coordinates. Defaults to 'absolute' */
  imageCropsMode?: 'absolute' | 'relative',

  /** If inference on CPU is preferred (defaults to GPU/Metal, coreML might be supported soon) */
  forceCPU?: boolean,
}
export type SKTFLiteTensorResult = SKTFLiteSingleTensorResult[];

export type SKTFLiteSingleTensorResult = {
  /** The supposed shape of this tensor (e.g. (1,192,192,3)) */
  shape: number[],
  /** The data of this tensor */
  data: number[]
}

export type SKTFLiteTensorImageTestArgs = {
  /** Path to the file */
  file: string,
  size?: {width: number, height: number},
  relativeCrops?: [number, number, number, number],
  opaque?: boolean,
  scale?: number,
  backgroundColor?: string,
}

type TensorflowLiteType = {
  multiply(a: number, b: number): Promise<number>;
  /**
   * 
   * @param params 
   * @returns Array corresponding to input files, each entry contains the output array of tensors from the model, the data field is simply a number array, while the shape is stored in the shape field.
   */
  runModelWithFiles(params: SKTFLiteRunModelWithFilesArg): Promise<SKTFLiteTensorResult[]>

  getModelInfo(params: {
    /** The path to the tensorflow lite model */
    model: string
  }): Promise<SKTFLiteModelInfoType>

  tensorImageTest(params: SKTFLiteTensorImageTestArgs): Promise<string>;
};

const { TensorflowLite } = NativeModules;

export default TensorflowLite as TensorflowLiteType;
