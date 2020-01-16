import aiohttp
import asyncio
import uvicorn
from fastai import *
from fastai.vision import *
from io import BytesIO
from starlette.applications import Starlette
from starlette.middleware.cors import CORSMiddleware
from starlette.responses import HTMLResponse, JSONResponse
from starlette.staticfiles import StaticFiles

export_file_url = 'https://content.leoauri.com/models/tagger_v0.1.0.pkl'
export_file_name = 'tagger_v0.1.0.pkl'

classes = [
    'abstract',
    'acoustic',
    'alien',
    'ambiance',
    'ambience',
    'ambient',
    'analog',
    'anxious',
    'artificial',
    'atmo',
    'atmos',
    'atmosphere',
    'atmospheric',
    'background',
    'background-sound',
    'bass',
    'beat',
    'bird',
    'birds',
    'birdsong',
    'bpm',
    'calm',
    'car',
    'chord',
    'cinematic',
    'city',
    'click',
    'club',
    'computer',
    'creepy',
    'dance',
    'dark',
    'deep',
    'delay',
    'digital',
    'drama',
    'dramatic',
    'drone',
    'drop',
    'drum',
    'drums',
    'echo',
    'eerie',
    'effect',
    'electric',
    'electro',
    'electronic',
    'engine',
    'english',
    'experimental',
    'fear',
    'female',
    'field-recording',
    'film',
    'forest',
    'future',
    'futuristic',
    'fx',
    'game',
    'general-noise',
    'ghost',
    'girl',
    'glitch',
    'guitar',
    'haunted',
    'hit',
    'holland',
    'horror',
    'house',
    'human',
    'impact',
    'industrial',
    'insects',
    'intro',
    'kick',
    'lo-fi',
    'loop',
    'machine',
    'male',
    'mechanical',
    'melody',
    'metal',
    'metallic',
    'minimal',
    'mood',
    'motor',
    'movie',
    'music',
    'nature',
    'night',
    'nightmare',
    'noise',
    'pad',
    'people',
    'percussion',
    'phantom',
    'piano',
    'processed',
    'radio',
    'rave',
    'retro',
    'reverb',
    'rhythm',
    'robot',
    'sample',
    'scary',
    'sci-fi',
    'scifi',
    'sfx',
    'short',
    'sinister',
    'snare',
    'sound',
    'sound-design',
    'sound-effect',
    'soundeffect',
    'soundscape',
    'space',
    'spaceship',
    'speech',
    'spooky',
    'spring',
    'stereo',
    'strange',
    'street',
    'strings',
    'summer',
    'suspense',
    'synth',
    'synthesizer',
    'talk',
    'techno',
    'terrifying',
    'terror',
    'thrill',
    'traffic',
    'trailer',
    'trance',
    'video-game',
    'vocal',
    'voice',
    'water',
    'weird',
    'white-noise',
    'wind',
    'woman'
    ]

path = Path(__file__).parent

app = Starlette()
app.add_middleware(CORSMiddleware, allow_origins=['*'], allow_headers=['X-Requested-With', 'Content-Type'])
app.mount('/static', StaticFiles(directory='app/static'))


async def download_file(url, dest):
    if dest.exists(): return
    async with aiohttp.ClientSession() as session:
        async with session.get(url) as response:
            data = await response.read()
            with open(dest, 'wb') as f:
                f.write(data)


async def setup_learner():
    await download_file(export_file_url, path / export_file_name)
    try:
        learn = load_learner(path, export_file_name)
        return learn
    except RuntimeError as e:
        if len(e.args) > 0 and 'CPU-only machine' in e.args[0]:
            print(e)
            message = "\n\nThis model was trained with an old version of fastai and will not work in a CPU environment.\n\nPlease update the fastai library in your training environment and export your model again.\n\nSee instructions for 'Returning to work' at https://course.fast.ai."
            raise RuntimeError(message)
        else:
            raise


loop = asyncio.get_event_loop()
tasks = [asyncio.ensure_future(setup_learner())]
learn = loop.run_until_complete(asyncio.gather(*tasks))[0]
loop.close()


@app.route('/')
async def homepage(request):
    html_file = path / 'view' / 'index.html'
    return HTMLResponse(html_file.open().read())


@app.route('/analyze', methods=['POST'])
async def analyze(request):
    img_data = await request.form()
    img_bytes = await (img_data['file'].read())
    img = open_image(BytesIO(img_bytes))
    
    prediction = learn.predict(img)[2]
    pred_tags = [class_name for i, class_name in enumerate(classes) if prediction[i] > 0.2]

    # TODO: take argmax prediction if none pass threshold
    # if not pred_tags:
    #     pred_tags = 

    return JSONResponse({'result': str(pred_tags)})


if __name__ == '__main__':
    if 'serve' in sys.argv:
        uvicorn.run(app=app, host='0.0.0.0', port=5000, log_level="info")
