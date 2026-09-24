from tests.baselines.official import qualify_choices
from baselines.rap.service import RAPService


if __name__ == '__main__':
    qualify_choices.GenerationService = RAPService
    qualify_choices.main()
