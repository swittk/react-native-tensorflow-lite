import { NativeModules } from 'react-native';

export type SKTFLiteRunModelWithFilesArg = {
  /** The path to the tensorflow lite model to use */
  model: string,
  /** The paths to the files to evaluate*/
  files: string[],
  /** The shapes of the imput tesors. CAREFUL THIS WILL CRASH YOUR APP IF NOT PROPERLY DEFINED
   * If this is not provided, the shapes are used from the model anyway.
   */
  shapes?: number[][]
}
export type SKTFLiteTensorResult = SKTFLiteSingleTensorResult[];

export type SKTFLiteSingleTensorResult = {
  /** The supposed shape of this tensor (e.g. (1,192,192,3)) */
  shape: number[],
  /** The data of this tensor */
  data: number[]
}

type TensorflowLiteType = {
  multiply(a: number, b: number): Promise<number>;
  /**
   * 
   * @param params 
   * @returns Array corresponding to input files, each entry contains the output array of tensors from the model, the data field is simply a number array, while the shape is stored in the shape field.
   */
  runModelWithFiles(params: SKTFLiteRunModelWithFilesArg): Promise<SKTFLiteTensorResult[]>
};

const { TensorflowLite } = NativeModules;

export default TensorflowLite as TensorflowLiteType;
