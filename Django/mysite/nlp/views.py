from django.shortcuts import render
from django.http import HttpResponse, HttpResponseNotFound
from nlp import model_predict

reviews = { 0: {'review' : "Bromwell High is a cartoon comedy. It ran at the same time as some other programs about school life, such as Teachers. My 35 years in the teaching profession lead me to believe that Bromwell High's satire is much closer to reality than is Teachers. The scramble to survive financially, the insightful students who can see right through their pathetic teachers' pomp, the pettiness of the whole situation, all remind me of the schools I knew and their students. When I saw the episode in which a student repeatedly tried to burn down the school, I immediately recalled ......... at .......... High. A classic line: INSPECTOR: I'm here to sack one of your teachers. STUDENT: Welcome to Bromwell High. I expect that many adults of my age think that Bromwell High is far fetched. What a pity that it isn't!",
               'polarity' : 'Positive',
               'rating' : 9},
            1: {'review' : "If you like adult comedy cartoons, like South Park, then this is nearly a similar format about the small adventures of three teenage girls at Bromwell High. Keisha, Natella and Latrina have given exploding sweets and behaved like bitches, I think Keisha is a good leader. There are also small stories going on with the teachers of the school. There's the idiotic principal, Mr. Bip, the nervous Maths teacher and many others. The cast is also fantastic, Lenny Henry's Gina Yashere, EastEnders Chrissie Watts, Tracy-Ann Oberman, Smack The Pony's Doon Mackichan, Dead Ringers' Mark Perry and Blunder's Nina Conti. I didn't know this came from Canada, but it is very good. Very good!",
               'polarity' : 'Positive',
               'rating' : 7},
            2: {'review' : "Story of a man who has unnatural feelings for a pig. Starts out with a opening scene that is a terrific example of absurd comedy. A formal orchestra audience is turned into an insane, violent mob by the crazy chantings of it's singers. Unfortunately it stays absurd the WHOLE time with no general narrative eventually making it just too off putting. Even those from the era should be turned off. The cryptic dialogue would make Shakespeare seem easy to a third grader. On a technical level it's better than you might think with some good cinematography by future great Vilmos Zsigmond. Future stars Sally Kirkland and Frederic Forrest can be seen briefly.",
               'polarity' : 'Negative',
               'rating' : 3},
            3: {'review' : "Robert DeNiro plays the most unbelievably intelligent illiterate of all time. This movie is so wasteful of talent, it is truly disgusting. The script is unbelievable. The dialog is unbelievable. Jane Fonda's character is a caricature of herself, and not a funny one. The movie moves at a snail's pace, is photographed in an ill-advised manner, and is insufferably preachy. It also plugs in every cliche in the book. Swoozie Kurtz is excellent in a supporting role, but so what?<br /><br />Equally annoying is this new IMDB rule of requiring ten lines for every review. When a movie is this worthless, it doesn't require ten lines of text to let other readers know that it is a waste of time and tape. Avoid this movie.",
                'polarity' : 'Negative',
                'rating' : 1},}

def index(request):
    return render(request, 'nlp/home.html', {'reviews': reviews})

def review(request):
    print(request.method, model_predict.result['polarity'])
    return render(request, 'nlp/review.html', {'title': 'Рецензия'})



def pageNotFound(request, exception):
    return HttpResponseNotFound('<h1>Страница не найдена</h1>')