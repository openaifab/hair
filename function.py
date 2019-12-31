import deepllabv3plus, google_images_download
import requests, cv2, torch, urllib.request
import pandas as pd
from PIL import Image
from trimap import trimap 
from pytorch_deep_image_matting.deep_image_matting import model_dim_fn, matting_result
cuda = torch.cuda.is_available()
print("cuda: " + str(cuda))
deep_image_matting_model = model_dim_fn(cuda)
print("matting model loading")


def google_img(search_name, output_folder, num=10, download=False):
    response = google_images_download.googleimagesdownload()
    if download:
        no_download = False
    else:
        no_download = True        
    data = response.download({
        "keywords": search_name,
        "color_type": "full-color",
        "format": "jpg",
        "size": "medium",
        "limit": num,
        'no_directory': True,
        'no_download': no_download,
        'silent_mode': True,
        "output_directory": output_folder})
    data = list(map(lambda x: x['image_link'], data[2]))
    return {search_name:data}

def flickr_img(search_name, output_folder, num=10, download=False):
    url_flickr = "https://www.flickr.com/"
    data_flickr = requests.get(url_flickr)
    api_key = str(data_flickr.content).split("api.site_key")[1]
    api_key = api_key.split('"')[1]
    url = "https://api.flickr.com/services/rest?sort=relevance&parse_tags=1&content_type=7&extras=can_comment%2Ccount_comments%2Ccount_faves%2Cdescription%2Cisfavorite%2Clicense%2Cmedia%2Cneeds_interstitial%2Cowner_name%2Cpath_alias%2Crealname%2Crotation%2Curl_m&per_page=" + str(num) +"&page=1&lang=zh-Hant-HK&text=" + search_name + "&viewerNSID=&method=flickr.photos.search&csrf=&api_key=" + api_key + "&format=json&hermes=1&hermesClient=1&reqId=3405dc98&nojsoncallback=1"
    data = requests.get(url)
    data = data.json()
    data = pd.DataFrame.from_dict(data['photos']['photo'])
    data = data.filter(["height_m", "width_m", "url_m"]) 
    data = data.fillna(0)
    data['height_m'] = data['height_m'].astype(int)
    data['width_m'] = data['width_m'].astype(int)
    data = data[data['height_m'] <= 500]
    data = data[data['height_m'] >0]
    data = data[data['width_m'] <= 500]
    data = data[data['width_m'] >0]
    data = data.reset_index(drop=True)
    if len(data)>0:
        if download:
            for i in range(len(data)):
                urllib.request.urlretrieve(data.url_m[i], output_folder + "/" + str(i+1) + ". " +data.url_m[i].split("/")[-1])
        return {search_name:list(data.url_m)}
    else:
        return {search_name:"no data"}
        
def seg_img(photo_input, seq_file, mask_file, ouput_folder, show, save):
    result = deepllabv3plus.run_deeplabv3plus(photo_input, seq_file, mask_file, ouput_folder, show, save)
    if show:
        return result
    
def trimap_output(original_input, mask_input, output_folder, output_name, 
                  trimap_save = True, seg_show = True, seg_save = True):
    mask = cv2.imread(mask_input, cv2.IMREAD_GRAYSCALE)
    trimap_01_01 = trimap(mask, size=1, erosion=1)
    trimap_25_05 = trimap(mask, size=25, erosion=5)
    trimap_30_10 = trimap(mask, size=30, erosion=10)
    trimap_30_20 = trimap(mask, size=30, erosion=20)
    trimap_40_20 = trimap(mask, size=40, erosion=20)
   
    matting_01_01 = matting_result(original_input, trimap_01_01, deep_image_matting_model, cuda)
    print("generate matting(size: 01, erosion: 01)")
    matting_25_05 = matting_result(original_input, trimap_25_05, deep_image_matting_model, cuda)
    print("generate matting(size: 25, erosion: 05)")
    matting_30_10 = matting_result(original_input, trimap_30_10, deep_image_matting_model, cuda)
    print("generate matting(size: 30, erosion: 10)")
    matting_30_20 = matting_result(original_input, trimap_30_20, deep_image_matting_model, cuda)
    print("generate matting(size: 30, erosion: 20)")
    matting_40_20 = matting_result(original_input, trimap_40_20, deep_image_matting_model, cuda)
    print("generate matting(size: 40, erosion: 20)")
    
    if trimap_save :
        Image.fromarray(trimap_01_01.astype('uint8')).save(output_folder + "/" + output_name + "_trimap_01_01.png")
        Image.fromarray(trimap_25_05.astype('uint8')).save(output_folder + "/" + output_name + "_trimap_25_05.png")
        Image.fromarray(trimap_30_10.astype('uint8')).save(output_folder + "/" + output_name + "_trimap_30_10.png")
        Image.fromarray(trimap_30_20.astype('uint8')).save(output_folder + "/" + output_name + "_trimap_30_20.png")
        Image.fromarray(trimap_40_20.astype('uint8')).save(output_folder + "/" + output_name + "_trimap_40_20.png")
    
    if seg_save:
        matting_01_01.save(output_folder + "/" + output_name + "_matting_01_01.png")
        matting_25_05.save(output_folder + "/" + output_name + "_matting_25_05.png")
        matting_30_10.save(output_folder + "/" + output_name + "_matting_30_10.png")
        matting_30_20.save(output_folder + "/" + output_name + "_matting_30_20.png")
        matting_40_20.save(output_folder + "/" + output_name + "_matting_40_20.png")
    
    if seg_show:
        return {"01_01":matting_01_01, 
                "25_05":matting_25_05, 
                "30_10":matting_30_10, 
                "30_20":matting_30_20, 
                "40_20":matting_40_20}