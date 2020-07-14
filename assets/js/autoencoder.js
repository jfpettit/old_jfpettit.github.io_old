const tf = require("@tensorflow/tfjs")

/**
 * The encoder portion of the model.
 *
 * @param {object} opts encoder configuration, includnig the following fields:
 *   - originaDim {number} Length of the input flattened image.
 *   - intermediateDim {number} Number of units of the intermediate (i.e.,
 *     hidden) dense layer.
 *   - latentDim {number} Dimensionality of the latent space (i.e,. z-space).
 * @param {number} opts.originalDim number of dimensions in the original data.
 * @param {number} opts.intermediateDim number of dimensions in the bottleneck.
 * @param {number} opts.latentDim number of dimensions in latent space.
 * @returns {tf.LayersModel} the encoder model.
 */
function encoder(opts) {
    const {originalDim, intermediateDim, latentDim} = opts;
  
    const inputs = tf.input({shape: [originalDim], name: 'encoder_input'});
    const x = tf.layers.dense({units: intermediateDim, activation: 'relu'})
                  .apply(inputs);
    const latent = tf.layers.dense({units: latentDim, name: 'z_mean'}).apply(x);
  
    const enc = tf.model({
      inputs: inputs,
      outputs: [latent],
      name: 'encoder',
    });
  
    // console.log('Encoder Summary');
    // enc.summary();
    return enc;
  }
  
  /**
   * This layer implements the 'reparameterization trick' described in
   * https://blog.keras.io/building-autoencoders-in-keras.html.
   *
   * The implementation is in the call method.
   * Instead of sampling from Q(z|X):
   *    sample epsilon = N(0,I)
   *    z = z_mean + sqrt(var) * epsilon
   */
  
  /**
   * The decoder portion of the model.
   *
   * @param {*} opts decoder configuration
   * @param {number} opts.originalDim number of dimensions in the original data
   * @param {number} opts.intermediateDim number of dimensions in the bottleneck
   *                                      of the encoder
   * @param {number} opts.latentDim number of dimensions in latent space
   */
  function decoder(opts) {
    const {originalDim, intermediateDim, latentDim} = opts;
  
    // The decoder model has a linear topology and hence could be constructed
    // with `tf.sequential()`. But we use the functional-model API (i.e.,
    // `tf.model()`) here nonetheless, for consistency with the encoder model
    // (see `encoder()` above).
    const input = tf.input({shape: [latentDim]});
    let y = tf.layers.dense({
      units: intermediateDim,
      activation: 'relu'
    }).apply(input);
    y = tf.layers.dense({
      units: originalDim,
      activation: 'sigmoid'
    }).apply(y);
    const dec = tf.model({inputs: input, outputs: y});
  
    // console.log('Decoder Summary');
    // dec.summary();
    return dec;
  }
  
  /**
   * The combined encoder-decoder pipeline.
   *
   * @param {tf.Model} encoder
   * @param {tf.Model} decoder
   *
   * @returns {tf.Model} the vae.
   */
  function ae(encoder, decoder) {
    const inputs = encoder.inputs;
    const encoderOutputs = encoder.apply(inputs);
    const encoded = encoderOutputs[2];
    const decoderOutput = decoder.apply(encoded);
    const v = tf.model({
      inputs: inputs,
      outputs: [decoderOutput, ...encoderOutputs],
      name: 'ae_mlp',
    })
  
    // console.log('VAE Summary');
    // v.summary();
    return v;
  }
  

  function aeLoss(inputs, outputs) {
    return tf.tidy(() => {
        const originalDim = inputs.shape[1];
        const decoderOutput = outputs[0];
        const latent = outputs[1];
    
        // First we compute a 'reconstruction loss' terms. The goal of minimizing
        // tihs term is to make the model outputs match the input data.
        const reconstructionLoss =
            tf.losses.meanSquaredError(inputs, decoderOutput).mul(originalDim);
    
        return reconstructionLoss.mean();
      }); 
  }
  
  module.exports = {
    ae,
    encoder,
    decoder,
    aeLoss,
  }