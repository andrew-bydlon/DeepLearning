def myplot(history):
    PLOT = input("Would you like to plot loss and accuracy? [y/n]")
    if PLOT == 'y':
        import matplotlib.pyplot as plt
        history_dict = history.history
        loss_values = history_dict['loss']
        val_loss_values = history_dict['val_loss']

        epochs = range(1, 21)

        plt.plot(epochs, loss_values, 'bo', label='Training Loss')
        plt.plot(epochs, val_loss_values, 'b', label='Validation Loss')
        plt.title('Training and validation loss')
        plt.xlabel("Epochs")
        plt.ylabel("Loss")
        plt.legend()

        plt.show()

        # plotting the training and validation accuracy

        plt.clf()
        acc_values = history_dict['accuracy']
        val_acc_values = history_dict['val_accuracy']

        plt.plot(epochs, acc_values, 'bo', label='Training Acc')
        plt.plot(epochs, val_acc_values, 'b', label='Validation Acc')
        plt.title('Training and Validation Accuracy')
        plt.xlabel("Epochs")
        plt.ylabel("Loss")
        plt.legend()

        plt.show()