import random
from gliner import GLiNER


class CaptionProcessor:
    def __init__(self):
        self.ethnic_labels = {
            'male': ['Latino male', 'Black male', 'Asian male', 'White male', 'Middle Eastern male', 'Indigenous male',
                     'Pacific Islander male', 'Mixed race male',
                     'Afro-Latino male', 'South Asian male', 'Southeast Asian male', 'East Asian male', 'Biracial male',
                     'Multiracial male', 'Arab male'],
            'female': ['Latina female', 'Black female', 'Asian female', 'White female', 'Middle Eastern female',
                       'Indigenous female', 'Pacific Islander female', 'Mixed race female',
                       'Afro-Latina female', 'South Asian female', 'Southeast Asian female', 'East Asian female',
                       'Biracial female', 'Multiracial female', 'Arab female'],
            'person': ['Latino person', 'Black person', 'Asian person', 'White person', 'Middle Eastern person',
                       'Indigenous person', 'Pacific Islander person', 'Mixed race person',
                       'Afro-Latino person', 'South Asian person', 'Southeast Asian person', 'East Asian person',
                       'Biracial person', 'Multiracial person', 'Arab person'],
            'man': ['Latino man', 'Black man', 'Asian man', 'White man', 'Middle Eastern man', 'Indigenous man',
                     'Pacific Islander man', 'Mixed race man',
                     'Afro-Latino man', 'South Asian man', 'Southeast Asian man', 'East Asian man', 'Biracial man',
                     'Multiracial man', 'Arab man'],
            'woman': ['Latino woman', 'Black woman', 'Asian woman', 'White woman', 'Middle Eastern woman', 'Indigenous woman',
                    'Pacific Islander woman', 'Mixed race woman',
                    'Afro-Latino woman', 'South Asian woman', 'Southeast Asian woman', 'East Asian woman', 'Biracial woman',
                    'Multiracial woman', 'Arab woman'],
            'people': ['people', 'people of different races'],
        }
        self.model = GLiNER.from_pretrained("urchade/gliner_largev2")

    def add_ethnic_labels(self, prompt: str, p: float = 0.8) -> str:
        words = prompt.split()
        for i, word in enumerate(words):
            if word in self.ethnic_labels and random.random() < p:
                words[i] = random.choice(self.ethnic_labels[word])

        modified_prompt = ' '.join(words)
        return modified_prompt

    def contains_person(self, prompt: str) -> bool:
        labels = ["first name", "last name"]
        entities = self.model.predict_entities(prompt, labels)
        return len(entities) > 0
