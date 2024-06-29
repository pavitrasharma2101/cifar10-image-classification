def main():
    print('Train samples are loading.')
    train_datas, train_labels = load_train_samples()
    print('Train samples are loaded.')
    print('Training:')
    train(datas=train_datas, labels=train_labels)
    print('Trained:')
    print('Test samples are loading.')
    test_datas, test_labels = load_test_samples()
    print('Testing:')
    test(datas=test_datas, labels=test_labels)
    print('Tested:')

if __name__ == '__main__':
    main()
