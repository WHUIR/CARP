import math


def get_test_list_mask(batch_size, test_rating, user_reviews, item_reviews, user_masks, item_masks):
    user_test_batchs, item_test_batchs, user_input_test_batchs, item_input_test_batchs, user_mask_test_batchs, item_mask_test_batchs, rating_input_test_batchs = [], [], [], [], [], [], []
    for count in range(int(math.ceil(len(test_rating) / float(batch_size)))):
        user_test, item_test, user_input_test, item_input_test, user_mask_test, item_mask_test, rating_input_test = [], [], [], [], [], [], []
        for idx in range(batch_size):
            index = (count * batch_size + idx)
            if (index >= len(test_rating)):
                break
            rating = test_rating[index]
            user_test.append(rating[0])
            item_test.append(rating[1])
            user_input_test.append(user_reviews.get(rating[0]))
            item_input_test.append(item_reviews.get(rating[1]))
            user_mask_test.append(user_masks.get(rating[0]))
            item_mask_test.append(item_masks.get(rating[1]))
            rating_input_test.append([rating[2]])
        user_test_batchs.append(user_test)
        item_test_batchs.append(item_test)
        user_input_test_batchs.append(user_input_test)
        item_input_test_batchs.append(item_input_test)
        user_mask_test_batchs.append(user_mask_test)
        item_mask_test_batchs.append(item_mask_test)
        rating_input_test_batchs.append(rating_input_test)
        #print count, len(item_input_test_batchs[count])
    return user_test_batchs, item_test_batchs, user_input_test_batchs, item_input_test_batchs, user_mask_test_batchs, item_mask_test_batchs, rating_input_test_batchs
