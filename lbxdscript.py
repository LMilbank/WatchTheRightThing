import pandas as pd
import os
import numpy as np 
import subprocess
from timeit import default_timer
import time
import pickle
import multiprocessing
from multiprocessing import Pool
from functools import partial

#Web stuff
from bs4 import BeautifulSoup
import requests
import asyncio
from aiohttp import ClientSession
import lxml
from get_user_ratings import get_user_data

#APIs
from tmdbv3api import TMDb
from tmdbv3api import Movie

#R
import rpy2
import rpy2.robjects as robjects
from rpy2.robjects import pandas2ri
from rpy2.robjects.packages import importr
from rpy2.robjects.conversion import localconverter















#Functions

    #Requests Functions

def get_or_create_eventloop():
    try:
        return asyncio.get_event_loop()
    except RuntimeError as ex:
        if "There is no current event loop in thread" in str(ex):
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            return asyncio.get_event_loop()
def fetch_async(urls, array):
    start_time = default_timer()
    loop = get_or_create_eventloop() 
    future = asyncio.ensure_future(fetch_all(urls, array)) 
    loop.run_until_complete(future) 
    tot_elapsed = default_timer() - start_time
    print('Total time taken : ' + str(tot_elapsed))
    return array
async def fetch_all(urls, array):
    tasks = []
    fetch.start_time = dict() 
    async with ClientSession() as session:
        for url in urls:
            task = asyncio.ensure_future(fetch(url, session, array, urls))
            tasks.append(task) 
        _ = await asyncio.gather(*tasks) 
async def fetch(url, session, array, urls):
    fetch.start_time[url] = default_timer()
    async with session.get(url) as response:
        r = await response.read()
        elapsed = default_timer() - fetch.start_time[url]
        print(url +  ' took ' +  str(elapsed))
        index = urls.index(url)
        array[index] = r
        return r, array


def reconcile_new_data(prev, new, tpe):
    '''Takes two paths to csvs of ratings, finds the new movies or movies with changed ratings (the delta) and 
    returns this in csv format  '''
    # Returns csv of new values to populate

    try:
        prevdf = pd.read_csv(prev,header = 0)
        newdf = pd.read_csv(new,header = 0)
        for i in range(len(prevdf)):
            found = 0
            j = 0
            #Type r = ratings w = watchlist
            if tpe == 'r':
                while j < len(newdf) and found!=1:
                    if prevdf.at[i, 'Letterboxd URI'] == newdf.at[j, 'Letterboxd URI'] and prevdf.at[i, 'Rating'] == newdf.at[j, 'Rating']:
                        newdf.drop(labels = j,inplace = True)
                        newdf.reset_index(drop = True, inplace = True)
                        found = 1
                    elif prevdf.at[i, 'Letterboxd URI'] == newdf.at[j, 'Letterboxd URI']:
                        prevdf.drop(labels = i, inplace = True)
                    j+=1
            if tpe == 'w':
                while j < len(newdf) and found!=1:
                    if prevdf.at[i, 'Letterboxd URI'] == newdf.at[j, 'Letterboxd URI']:
                        newdf.drop(labels = j,inplace = True)
                        newdf.reset_index(drop = True, inplace = True)
                        found = 1
                    j+=1

        size = len(new)
        ratingspathmod = new [:size-4] 
        ratings_path = ratingspathmod + 'delta.csv'
        newdf.to_csv(ratings_path)
        prevdf.to_csv(prev)
        return ratings_path
    except: 
        return new


def get_data_ratings(path, gl = False):
    ''' This function uses the TMDB api to fill out data on a movie based on its Letterboxd link
    paramater gl is to be applied if the list is a general list not a users watchlist or ratings'''
    
    #Initialise TMDB 
    tmdb = TMDb()
    tmdb.api_key = '128ce08c426930b2efff71e8634bdcac'

    g1list = []
    g2list = []

    #Read in ratings data 
    if gl == True:
        df = pd.read_csv(path,header = 2)
    else:
        df = pd.read_csv(path,header = 0)
    rating_file_name = os.path.basename(path)

    #Find urls and fetch the responses
    urls = ["" for x in range(len(df))]
    for i in range(len(df)):
        try:
            urls[i] = (df.loc[i, 'Letterboxd URI'])
        except:
            urls[i] = df.loc[i, 'URL']
    responses = fetch_async(urls, ["" for x in range(len(df))])

    for i in range(len(df)):
        #Parse the response 
        soup = BeautifulSoup(responses[i], 'lxml')

        #Grab the average rating, director and year of release
        avgr = soup.find('meta', attrs={'name':'twitter:data2'})
        drctr = soup.find('meta', attrs = {'name' : 'twitter:data1'})
        year = soup.find('meta', attrs={'name' : 'twitter:title'})

        #Grab the part of site containing movie id 
        runtime = soup.find('p', attrs = {'class' : 'text-link text-footer'})

        #Load these values in to the dataframe 
        if year!=None:
            df.at[i, 'Year'] = int(year['content'][-5:-1])
        if avgr!=None:
            df.at[i,'Average Rating'] = float(avgr['content'][0:4])
        if drctr!=None:
            df.at[i, 'Director'] = drctr['content']
        
        #Pull out the movie id from the link
        try: 
            tmdblnk = runtime.contents[3].attrs['href']
            idn = []
            for j in range(len(tmdblnk)):
                if  tmdblnk[j].isdigit():
                    idn.append(tmdblnk[j])
            strings = [str(integer) for integer in idn]
            a_string = "".join(strings)
            idn = int(a_string)            
            df.at[i, 'Name'] = idn

        #Initialise the movie instance and grab the details
        
            movie = Movie()
            mdvalues = movie.details(movie_id = idn)

            if int(mdvalues.release_date[0:4]) != int(df.at[i, 'Year']):
                a = int(mdvalues.release_date[0:4])
                b = int(df.at[i, 'Year'])
                c = a - b
                if abs(c)>2:
                    df.drop(labels = i,inplace = True)

            #If non zero values add:
            #                       budget 
            #                       revenue
            #                       language 
            #                       runtime
            #                       genre (2 if available)
            else:
                try:
                    if hasattr(mdvalues, 'budget'):
                        if mdvalues.budget != 0:
                            df.at[i, 'Budget'] = mdvalues.budget
                    if hasattr(mdvalues, 'revenue'):
                        if mdvalues.revenue != 0:
                            df.at[i, 'Revenue'] = mdvalues.revenue
                    if hasattr(mdvalues, 'original_language'):
                        if mdvalues.original_language != 'en':
                            df.at[i, 'Language'] = 'Non-English'
                        else:
                            df.at[i, 'Language'] = 'English'
                    if hasattr(mdvalues, 'runtime'):
                        if mdvalues.runtime != 0:
                            df.at[i, 'Runtime'] = mdvalues.runtime
                    if hasattr(mdvalues, 'genres'):
                        df.at[i, 'Genre_1'] = mdvalues.genres[0]['name']
                        g1list.append(mdvalues.genres[0]['name'])

                        if len(mdvalues.genres) > 1:
                            df.at[i, 'Genre_2'] = mdvalues.genres[1]['name']
                            g2list.append(mdvalues.genres[1]['name'])
                    

                except:
                    pass
        except:
            df.drop(labels = i,inplace = True)


    #Save out the final now populated dataframe 
    final_ratings_df = df
    size = len(path)
    ratingspathmod = path [:size-4] 
    ratings_path = ratingspathmod + 'output.csv'
    final_ratings_df.to_csv(ratings_path)

    return ratings_path

def add_delta(ogpath, deltapath):
    ''' Combines existing user data with new user data '''
    ogdf = pd.read_csv(ogpath,header = 0)
    deldf = pd.read_csv(deltapath,header = 0)    
    outputdf = pd.concat([ogdf, deldf])
    outputdf.reset_index(drop = True, inplace = True)
    outputdf.drop(outputdf.columns[outputdf.columns.str.contains('unnamed',case = False)],axis = 1, inplace = True)
    outputdf.to_csv(ogpath)
    return ogpath


def pathwithoutextension(path):
    ''' Removes .xxx from a filepath '''
    size = len(path)
    pathmod = path [:size-4]
    return pathmod



def fit_rpart2(username):
    '''
    Fits a classification tree model using RPART through RPy2 on ALREADY POPULATED csvs
    '''
    
    #R imports
    start_time = default_timer()
    rprint = robjects.globalenv.find("print")
    utils = importr('utils')
    rpart = importr('rpart')
    rattle = importr('rattle')
    rpartplot = importr('rpart.plot')
    stats = importr('stats')
    base = importr('base')
    tot_elapsed = default_timer() - start_time



    #Reading in the data
    dataf = utils.read_csv('users/' + username + '/ratingsoutput.csv')
    tot_elapsed2 = default_timer() - start_time


    #Fit the model
    fit = rpart.rpart('Rating ~ Average.Rating + Year + Language + Genre_1 + Genre_2 + Runtime + Budget + Revenue', data = dataf, method ="class")
    tot_elapsed3 = default_timer() - start_time


    with localconverter(robjects.default_converter + pandas2ri.converter):
        pd_frame = robjects.conversion.rpy2py(dataf)

    #Gather parameters for users model
    g1list =[]
    g2list = []
    pd_frame.reset_index(inplace = True)
    for i in range(len(pd_frame)):
        if pd_frame.at[i, 'Genre_1'] not in g1list:
            g1list.append(pd_frame.at[i,'Genre_1'])
        if pd_frame.at[i, 'Genre_2'] not in g2list:
            g2list.append(pd_frame.at[i,'Genre_2'])
    watched = pd_frame['Name']


    #Make/save pdf of tree model
    grdevices = importr('grDevices')
    grdevices.pdf(file="fancytest.pdf", width=11.75, height=8.25)
    rattle.fancyRpartPlot(fit)
    grdevices.dev_off()


    return fit, g1list, g2list, watched

def predict_rpart2(username, csvpath,rarray):
    """
    This function takes an existing rpart model and uses it to 
    predict expected ratings for a list of films and return recomendations.

    Parameters
    ----------
    username : string
        username of desired user
    csvpath : string
        filepath to list for predictions
    rarray : array
        array of imported r packages

    Returns
    -------
    posterarray : array
        array of urls to the recommended film's posters
    rec : array
        array of TMDB id numbers for the recommended films

    """

    # Initialise timer
    start_time = default_timer()

    # Load the R packages
    utils = rarray[0]
    rpart = rarray[1]
    rattle = rarray[2]
    rpartplot = rarray[3]
    stats = rarray[4]
    base = rarray[5]
    car = rarray[6]


    print(str(default_timer() - start_time) + '  R Imports Completed')

    # Load database and list csv
    f = open("lbxd.pickle", 'rb+')
    userbase = pickle.load(f)
    model = userbase[username][0] 
    g1 = userbase[username][1] 
    g2 = userbase[username][2]
    watched = userbase[username][3]
    films = pd.read_csv(csvpath)

    print(str(default_timer() - start_time) + '  Userbase and Film Sheet Loaded')

    #Prepare films list for predictions
    films = parallelize_dataframe(films, g1, g2, watched, clean_list_for_user)
    films.rename(columns = {'Average Rating': 'Average.Rating'}, inplace = True)
    films.reset_index(drop = True)
    final_watchlist_df = films
    
    print(str(default_timer() - start_time) + '  Glists, watched, rename')

    with localconverter(robjects.default_converter + pandas2ri.converter):
        films = robjects.conversion.py2rpy(films)

    filmnames = base.match('Name', films.colnames)
    filmnames = films.rx2(filmnames)
    films.rownames = filmnames

    # Generate predictions

    predictions = robjects.r.predict(model,films,method = 'class')
    print(str(default_timer() - start_time) + '  R manipulation and predictions')

    # Convert the predictions to pandas df
    with localconverter(robjects.default_converter + pandas2ri.converter):
        predf = robjects.conversion.rpy2py(predictions)

    predf = pd.DataFrame(list(map(np.ravel, predf)))

    #Setting up for calcs

    names = np.asarray(filmnames)
    predf.rename(columns = {0: '0.5', 1:'1', 2 : '1.5', 3:'2', 4:'2.5', 5:'3', 6:'3.5',7:'4', 8:'4.5',9:'5'}, inplace = True)
    predf['Name'] = names

    print(str(default_timer() - start_time) + '  Naming and reconversion to pandas')
    
    finaldf = predf
    finaldf['Genre_1'] = final_watchlist_df['Genre_1']
    finaldf['Language'] = final_watchlist_df['Language']
    finaldf['Runtime'] = final_watchlist_df['Runtime']


    vals =['0.5','1','1.5','2','2.5','3','3.5','4','4.5','5']
    avgpred = []
    for i in range(len(finaldf)):
        avgr = 0
        for j in range(len(vals)):
            a = finaldf.at[i, vals[j]]
            
            avgr = avgr + float(vals[j]) * a
        finaldf.at[i, 'AvgRating'] = avgr


    maxind = finaldf.sort_values('AvgRating', ascending = False)
    maxind = maxind.reset_index()
    print(str(default_timer() - start_time) + '  Setting up finaldf')

    #Let's grab some reccomendations 
    rec = []

    #Highest predicted rating
    rec.append(maxind.at[0, 'Name'])

    #Next highest but different genre
    count  = 1
    nope = 0
    while nope != 1 :
        
        if maxind.at[count, 'Genre_1'] != maxind.at[0, 'Genre_1'] :
            rec.append(maxind.at[count, 'Name'])
            nope = 1
        count +=1
    #Next highest but different language 
    holdcount = count - 1 
    count  = 1
    nope = 0
    try:
        while nope != 1 :
            if maxind.at[count, 'Language'] != maxind.at[0, 'Language'] and count!= holdcount :
                rec.append(maxind.at[count, 'Name'])
                nope = 1
            count +=1
    except:
        count = 1
        while nope != 1 :
            if count!= holdcount :
                rec.append(maxind.at[count, 'Name'])
                nope = 1
            count +=1
    holdcount2 = count - 1

    #Now shorter movies
    count = 1 
    nope = 0
    try : 
        while nope != 2:
            if maxind.at[count, 'Runtime'] < 100 and count != holdcount and count != holdcount2:
                rec.append(maxind.at[count, 'Name'])
                nope += 1
            count+=1
    except: 
        print('No short movies')
    print(str(default_timer() - start_time) + '  Recs Found')

    #Grab the poster URLs

    films = []
    tmdb = TMDb()
    tmdb.api_key = '128ce08c426930b2efff71e8634bdcac'
    CONFIG_PATTERN = 'http://api.themoviedb.org/3/configuration?api_key={key}'
    KEY = '128ce08c426930b2efff71e8634bdcac'
    url = CONFIG_PATTERN.format(key=KEY)
    r = requests.get(url)
    config = r.json()
    base_url = config['images']['base_url']
    max_size = 'original'
    posterarray = []
    for i in range(len(rec)):
        movie = Movie()
        providerarray = []
        
        try:
            mdvalues = movie.details(movie_id = rec[i])
            films.append(mdvalues.title)
            posterpath = mdvalues.poster_path
            posterurl = base_url + max_size + posterpath
            posterarray.append(posterurl)
        except:
            films.append('rec failed')
    print(str(default_timer() - start_time) + 'Final time (rec finding)')
    return posterarray, rec

def parallelize_dataframe(df, g1, g2, watched, func, n_cores=4):
    """
    This function takes a given function and applies it to a dataframe 
    with a users parameters using multiprocessing.

    Parameters
    ----------
    df : pandas dataFrame
        dataframe of films
    g1 : array
        array of strings for each acceptable Genre1 value
    g2 : array
        array of strings for each acceptable Genre2 value
    watched : array
        array of strings for each film that has been seen 
    func : function
        function to be applied
    n_cores : int
        number of cores on device (defaults to 4)
    Returns
    -------
    df : pandas dataFrame
        modified dataFrame

    """
    #Establish partial function
    func = partial(func, g1, g2, watched)
    #Divide frame
    df_split = np.array_split(df, 2*n_cores)
    #Set up pool and apply function
    pool = Pool(2*n_cores)
    df = pd.concat(pool.map(func, df_split))
    pool.close()
    pool.join()
    return df

def clean_list_for_user(g1, g2, watched, films):
    ''' This function takes a users parameters and a list of films and removes unacceptable films
    (Of an unseen genre or films that have already been seen) '''
    watched = np.asarray(watched)
    for i in films.index:
        try:
            if np.isnan(films.at[i,'Genre_2']) :
                films.at[i, 'Genre_2'] = ''
        except:
            pass
        if films.at[i,'Genre_1'] not in g1:
            films.drop(labels = i,inplace = True)
        elif films.at[i,'Genre_2'] not in g2:
            films.drop(labels = i,inplace = True)
        elif films.at[i, 'Name'] in watched:

            films.drop(labels = i,inplace = True)
    return films
    
    


def update_users():
    ''' This function updates the models and stored parameters for all existing users'''
    f = open("lbxd.pickle", 'rb+')
    userbase = pickle.load(f)
    print(userbase)
    
    for username in userbase:
        get_user_data(username)
        deltapath = reconcile_new_data('users/' + username + '/ratingsoutput.csv', 'users/' + username + '/ratings.csv', 'r' )
        if deltapath == 'users/' + username + '/ratings.csv':
            ratingspath = get_data_ratings(deltapath)
        else:
            deltaout = get_data_ratings(deltapath)
            ratingspath = add_delta('users/' + username + '/ratingsoutput.csv', deltaout)
        model, g1, g2, watched = fit_rpart2(username)
        userbase[username] = [model, g1, g2, watched]
    f.seek(0)
    f.truncate()
    pickle.dump(userbase, f)
    print(userbase)
    f.close()

def create_new_user(username):
    ''' This function takes a new username and scrapes Letterboxd to fit a model and populate the 
    parameter fields in the database.'''
    f = open("lbxd.pickle", 'rb+')
    userbase = pickle.load(f)
    get_user_data(username)
    get_data_ratings('users/' + username + '/ratings.csv')
    model, g1, g2, watched = fit_rpart2(username)
    userbase[username] = [model, g1, g2, watched]
    f.seek(0)
    f.truncate()
    pickle.dump(userbase, f)
    print(userbase)
    f.close()

def create_picklebase():
    '''This function initialised the pickle file with one empty user (me)'''
    diction = {'liammilbank' : [0,0,0]}
    file_to_write = open("lbxd.pickle", "wb")
    pickle.dump(diction, file_to_write)


if __name__ == '__main__':

    rarray = [importr('utils'),importr('rpart'), importr('rattle'), importr('rpart.plot'),importr('stats'),importr('base'),importr('car')]
    array, rec = predict_rpart2('liammilbank',"General_Lists/Famous.Lists/1.csv", rarray)
    print(array)
    print(rec)


    