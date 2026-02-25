


######################################################################################################################################################
#############################################  Deep_Library_BCI details  #############################################
import tensorflow as tf

from tensorflow.keras import layers, models, losses, optimizers, metrics


class MultiTaskModel(tf.keras.Model):
    def __init__(
        self,
        input_shape = (128,1),
        latent_dim=16,
        conv_filters=(16, 64, 256, 128),
        kernel_size=3,
        activation_fn=layers.LeakyReLU,
        use_decoder=True,
        use_classifier=True,
        latent_activation =  None, # use tanh if you want to compare with handextracted features ???
        **kwargs
    ):
        super().__init__(**kwargs)

        self.input_shape_ = input_shape
        self.latent_dim = latent_dim
        self.conv_filters = conv_filters
        self.kernel_size = kernel_size
        self.activation_fn = activation_fn
        self.latent_activation = latent_activation

        # Build components
        self.encoder = self._build_encoder()
        self.decoder = self._build_decoder() if use_decoder else None
        self.classifier = self._build_classifier() if use_classifier else None

    # --------------------------------------------------
    # ENCODER
    # --------------------------------------------------
    def _build_encoder(self):
        inputs = layers.Input(shape=self.input_shape_, name="encoder_input")

        x = inputs

        for n_layer in range(len(self.conv_filters)):
            filters = self.conv_filters[n_layer]
            x = layers.Conv1D(filters, self.kernel_size, padding="same")(x)
            x = layers.BatchNormalization()(x)            
            x = self.activation_fn()(x)
            x = layers.MaxPooling1D()(x)                        # downsample by 2 n_filters times (4-> shape is 128/16 = 8)
            if n_layer < len(self.conv_filters) -1 :
                x = layers.Dropout(0.3)(x)

        x = layers.Flatten()(x) # 8 * 128 = 1024

        # x = layers.Dense(self.latent_dim*2)(x) # TODO yes or no?? (Attenttion to batch norm here)
        #x = self.activation_fn(x)
        latent = layers.Dense(self.latent_dim, name="latent", activation=self.latent_activation)(x)

        return models.Model(inputs, latent, name="encoder")

    # --------------------------------------------------
    # DECODER
    # --------------------------------------------------
    def _build_decoder(self):
        if self.input_shape_[0] % (2 ** len(self.conv_filters)) != 0:
            raise ValueError("Input length must be divisible by total downsampling factor")
        
        latent_inputs = layers.Input(shape=(self.latent_dim,), name="decoder_input")

        downsample_factor = 2 ** len(self.conv_filters)
        h = self.input_shape_[0] // downsample_factor
        c = self.conv_filters[-1]
        print(h,c)
        x = layers.Dense(h * c)(latent_inputs)
        x = self.activation_fn()(x)
        x = layers.Reshape((h, c))(x)

        for filters in reversed(self.conv_filters):
            x = layers.UpSampling1D()(x)
            x = layers.Conv1D(filters, self.kernel_size, padding="same")(x)
            # x = layers.Conv1DTranspose(filters, self.kernel_size, strides=2, padding="same")(x)
            x = self.activation_fn()(x)

        outputs = layers.Conv1D(
            self.input_shape_[-1],
            kernel_size=3,
            padding="same",
            activation="linear",   # TODO attention to the normalization of data
            name="reconstruction"
        )(x)

        return models.Model(latent_inputs, outputs, name="decoder")

    # --------------------------------------------------
    # CLASSIFIER
    # --------------------------------------------------
    def _build_classifier(self):
        latent_inputs = layers.Input(shape=(self.latent_dim,), name="classifier_input")

        outputs = layers.Dense(
            1,
            activation="sigmoid",
            name="classification"
        )(latent_inputs)

        return models.Model(latent_inputs, outputs, name="classifier")

    # --------------------------------------------------
    # FORWARD PASS
    # --------------------------------------------------
    def call(self, inputs, training=False):

        latent = self.encoder(inputs, training=training)

        outputs = {}

        if self.decoder is not None:
            outputs["reconstruction"] = self.decoder(latent, training=training)

        if self.classifier is not None:
            outputs["classification"] = self.classifier(latent, training=training)

        # Return single tensor if only one head exists
        if len(outputs) == 1:
            return next(iter(outputs.values()))

        return outputs

    # --------------------------------------------------
    # COMPILE FOR DIFFERENT CASES
    # --------------------------------------------------
    def compile_cases(self, optimizer, loss_reconstruction=None, loss_classification=None,
                      loss_weights=None, metric_reconstruction=None, metric_classification=None):

        
        if loss_weights is None:
            loss_weights = {"reconstruction": 1.0, "classification": 1.0}
        if metric_reconstruction is None:
            metric_reconstruction = ["mae"]
        if metric_classification is None:
            metric_classification = ["accuracy"]
        
        # Classifier-only
        if self.decoder is None and self.classifier is not None:
            self.compile(optimizer=optimizer, loss=loss_classification, metrics=metric_classification)

        # Autoencoder-only
        elif self.decoder is not None and self.classifier is None:
            self.compile(optimizer=optimizer, loss=loss_reconstruction, metrics=metric_reconstruction)

        # Multi-output
        elif self.decoder is not None and self.classifier is not None:
            loss_fns = {"reconstruction": loss_reconstruction,
            "classification": loss_classification}
            metrics_dict= {"reconstruction": metric_reconstruction,
                           "classification": metric_classification}

            self.compile(optimizer=optimizer, loss=loss_fns, metrics=metrics_dict, loss_weights=loss_weights)

        else:
            raise RuntimeError("Model has neither decoder nor classifier.")
        

    # --------------------------------------------------
    # DEFINE FIT CASES
    # --------------------------------------------------

    def fit_cases(self, x_train, x_val, y_train=None, y_val=None, epochs=100, batch_size=64, **kwargs):
        
        # Classifier-only
        if self.decoder is None and self.classifier is not None:
            storia = self.fit(x_train, y_train, validation_data=(x_val, y_val), epochs=epochs, batch_size=batch_size, **kwargs)
            return storia

        # Autoencoder-only
        elif self.decoder is not None and self.classifier is None:
            storia = self.fit(x_train, x_train, validation_data=(x_val,x_val), epochs=epochs, batch_size=batch_size, **kwargs)
            return storia

        # Multi-output
        elif self.decoder is not None and self.classifier is not None:
            storia = self.fit(x_train, {"reconstruction": x_train, "classification": y_train}, validation_data=(x_val, {"reconstruction": x_val, "classification": y_val}), epochs=epochs, batch_size=batch_size, **kwargs)
            return storia

        else:
            raise RuntimeError("Model has neither decoder nor classifier.")  



